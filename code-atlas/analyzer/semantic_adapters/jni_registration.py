"""Evidence-first static JNI RegisterNatives discovery, no project code executed.

Recognizes a narrow but common, compiler-validated shape:
  - JNINativeMethod table with literal (name, JVM descriptor, function pointer)
  - jclass local initialized by FindClass("fully/qualified/Class")
  - RegisterNatives(class_local, table, count) in the same C/C++ function

The evidence is *candidate registration*, never evidence of dynamic loading or
successful runtime registration. Ambiguous or symbolic expressions are skipped.
The compiler AST is used to avoid searching comments / matching strings.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

MAX_FILES = 48
MAX_BYTES = 800_000
MAX_AST = 24_000_000


def _walk(node):
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get('inner', []):
        yield from _walk(child)


def _ref(node, kinds=('VarDecl', 'ParmVarDecl')):
    refs = [e.get('referencedDecl', {}).get('id') for e in _walk(node)
            if e.get('kind') == 'DeclRefExpr'
            and e.get('referencedDecl', {}).get('kind') in kinds]
    return refs[0] if len(refs) == 1 else None


def _literal(node):
    values = [e.get('value') for e in _walk(node) if e.get('kind') == 'StringLiteral']
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    try:
        result = json.loads(values[0])
    except (ValueError, TypeError):
        return None
    return result if isinstance(result, str) else None


def _called(node):
    if node.get('kind') not in ('CallExpr', 'CXXMemberCallExpr'):
        return None
    children = node.get('inner', [])
    if not children:
        return None
    members = [e.get('name') for e in _walk(children[0])
               if e.get('kind') == 'MemberExpr' and e.get('name')]
    if len(members) == 1:
        return members[0]
    refs = [e.get('referencedDecl', {}).get('name') for e in _walk(children[0])
            if e.get('kind') == 'DeclRefExpr'
            and e.get('referencedDecl', {}).get('kind') == 'FunctionDecl']
    return refs[0] if len(refs) == 1 else None


def _jni_headers():
    """Use JDK headers only when installed; never download or execute builds."""
    paths = []
    env = os.environ.get('JAVA_HOME')
    if env:
        paths.append(Path(env))
    javac = shutil.which('javac')
    if javac:
        paths.append(Path(javac).resolve().parent.parent)
    for home in paths:
        include = home / 'include'
        if (include / 'jni.h').is_file():
            args = ['-I', str(include)]
            for platform in ('linux', 'darwin', 'win32', 'freebsd'):
                if (include / platform).is_dir():
                    args += ['-I', str(include / platform)]
            return args
    return []


def _location(node, offsets, text):
    # Clang offsets are bytes. The file path is already fixed to this TU.
    loc = node.get('loc') or node.get('range', {}).get('begin') or {}
    if 'offset' not in loc:
        loc = node.get('range', {}).get('begin') or {}
    if 'offset' not in loc:
        return None
    offset = loc['offset']
    if not isinstance(offset, int) or offset < 0 or offset >= len(text):
        return None
    return bisect.bisect_right(offsets, offset)


def _table_entry(init):
    if init.get('kind') != 'InitListExpr' or len(init.get('inner', [])) != 3:
        return None
    name, signature, implementation = init['inner']
    name, signature = _literal(name), _literal(signature)
    function_ref = _ref(implementation, ('FunctionDecl',))
    if not name or not signature or not function_ref:
        return None
    if not _valid_descriptor(signature):
        return None
    return name, signature, function_ref


def _valid_descriptor(value):
    """Require one valid JVM method descriptor, not a free-form string."""
    if not isinstance(value, str) or not value.startswith('(') or len(value) > 1024:
        return False
    def typ(i, allow_void=False):
        if i >= len(value):return None
        c = value[i]
        if c in 'BCDFIJSZ' or (c == 'V' and allow_void):return i+1
        if c == '[':
            j = i
            while j < len(value) and value[j] == '[':
                j += 1
            if j >= len(value) or value[j] == 'V':return None
            return typ(j)
        if c == 'L':
            end = value.find(';',i+1)
            if end < 0 or end==i+1:return None
            inner = value[i+1:end]
            if not re.fullmatch(r'[A-Za-z0-9_$/]+',inner):return None
            return end+1
        return None
    i=1
    while i < len(value) and value[i] != ')':
        next_i = typ(i)
        if next_i is None:return False
        i=next_i
    if i >= len(value) or value[i] != ')':return False
    end=typ(i+1,True)
    return end==len(value)


def _extract_registrations(tree, path, raw):
    """Each registration: (owner, method, descriptor, function-ref, evidence-line).

    Only AST references with matching *declaration IDs* are accepted. This
    prevents coincidental matching of variables from other scopes.
    """
    offsets = [0]+[i+1 for i,b in enumerate(raw) if b==10]
    tables = {}
    functions = {}
    for n in _walk(tree):
        if n.get('kind') == 'FunctionDecl' and n.get('id') and n.get('name'):
            line = _location(n, offsets, raw)
            if line:
                functions[n['id']] = (n['name'],line)
        if n.get('kind') != 'VarDecl' or not n.get('id'):
            continue
        q = (n.get('type') or {}).get('qualType', '')
        if not re.search(r'\bJNINativeMethod\b', q) or '[' not in q:
            continue
        init = next((c for c in n.get('inner',[]) if c.get('kind')=='InitListExpr'), None)
        if not init:
            continue
        rows = [_table_entry(c) for c in init.get('inner',[])]
        if rows and all(r is not None for r in rows):
            tables[n['id']] = rows
    results=[]
    for fn in _walk(tree):
        if fn.get('kind')!='FunctionDecl' or not any(c.get('kind')=='CompoundStmt' for c in fn.get('inner', [])):
            continue
        classes={}
        # Work within one function and track explicit, type-checked locals.
        for n in _walk(fn):
            if n.get('kind') != 'VarDecl' or not n.get('id'):
                continue
            if (n.get('type') or {}).get('qualType') != 'jclass':
                continue
            expressions=[x for x in _walk(n) if _called(x)=='FindClass']
            if len(expressions)!=1:continue
            args=expressions[0].get('inner',[])[1:]
            if len(args)!=1:continue
            owner=_literal(args[0])
            if owner and re.fullmatch(r'[A-Za-z0-9_$/]+',owner):
                classes[n['id']]=owner
        # Only accept calls where the target class and table are direct variable
        # references. Complex control/data flows remain explicitly unresolved.
        for n in _walk(fn):
            if _called(n)!='RegisterNatives':continue
            children=n.get('inner',[])
            member = children[0]
            args=children[1:]
            if n.get('kind')=='CallExpr':
                # C-style (*env)->RegisterNatives(env, clazz, table, size)
                args=args[1:] if len(args)==4 else []
            if len(args)!=3:continue
            # A constant registration size bounds how many table entries can
            # possibly be registered. For symbolic sizes, retain candidates.
            count_expr = [v for v in _walk(args[2]) if v.get('kind') == 'IntegerLiteral']
            count_limit = None
            if len(count_expr) == 1 and len(args[2].get('inner',[])) == 0:
                try:count_limit=int(count_expr[0]['value'])
                except (KeyError,ValueError,TypeError):pass
            class_ref, table_ref=_ref(args[0]),_ref(args[1])
            owner=classes.get(class_ref)
            rows=tables.get(table_ref)
            line=_location(n, offsets, raw)
            if not owner or not rows or line is None:continue
            if count_limit is not None:
                if count_limit <= 0:continue
                rows=rows[:count_limit]
            for name,desc,func_ref in rows:
                if func_ref in functions:
                    func,func_line=functions[func_ref]
                    results.append((owner,name,desc,func,func_line,line))
    return results


def connect_register_natives(root:Path, candidates:dict, graphs:dict) -> list[dict]:
    """Enrich JVM native method graphs; no project-specific names or paths."""
    root=root.resolve()
    if not shutil.which('clang++'):
        return [{'reason':'jni_register_clang_unavailable'}]
    headers=_jni_headers()
    if not headers:
        return [{'reason':'jni_register_headers_unavailable'}]
    # Compiler-backed native declaration: exact JVM owner + descriptor required.
    declarations=defaultdict(list)
    for graph in graphs.values():
        if graph.get('parser') not in ('javac_ast','kotlin_psi'):continue
        for n in graph.get('nodes', []):
            if n.get('external') or not (n.get('native_declaration') or n.get('external_declaration')):continue
            if n.get('jvm_owner') and _valid_descriptor(n.get('jvm_descriptor','')):
                declarations[(n['jvm_owner'], n['name'], n['jvm_descriptor'])].append(n)
    if not declarations:
        return []
    paths=[]
    for path,item in candidates.items():
        if item.get('language') not in ('cpp','c') or Path(path).suffix.lower() not in ('.cpp','.cc','.cxx','.c'):continue
        f=(root/path).resolve()
        if not f.is_relative_to(root) or not f.is_file() or f.stat().st_size>MAX_BYTES:continue
        raw=f.read_bytes()
        if b'JNINativeMethod' in raw and b'RegisterNatives' in raw:
            paths.append((path,raw))
    warnings=[]
    for path,raw in sorted(paths)[:MAX_FILES]:
        suffix=Path(path).suffix.lower()
        lang='c' if suffix=='.c' else 'c++'
        cmd=[shutil.which('clang++'),'-x',lang, '-std=c11' if lang=='c' else '-std=c++17',
             '-fsyntax-only','-Wno-everything','-Xclang','-ast-dump=json',
             *headers,'-I',str(root),str(root/path)]
        try:
            proc=subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=20)
            if proc.returncode or not proc.stdout or len(proc.stdout)>MAX_AST:
                warnings.append({'path':path,'reason':'jni_registration_ast_incomplete'});continue
            tree=json.loads(proc.stdout)
        except (OSError,ValueError,subprocess.TimeoutExpired) as e:
            warnings.append({'path':path,'reason':'jni_registration_ast_failed','details':str(e)[:120]});continue
        native_graph=graphs.get(path)
        if not native_graph or native_graph.get('parser')!='clang_ast':
            warnings.append({'path':path,'reason':'jni_registration_native_graph_unavailable'});continue
        native_nodes=[n for n in native_graph['nodes'] if n.get('kind')=='function'
                      and n.get('has_body') and not n.get('external')]
        lines=raw.decode('utf8',errors='replace').splitlines()
        table_hash=hashlib.sha256(raw).hexdigest()
        for owner,name,desc,func,func_line,registration_line in _extract_registrations(tree,path,raw):
            targets=declarations.get((owner,name,desc),[])
            native_targets=[n for n in native_nodes if n.get('name')==func and n.get('line')==func_line]
            if len(targets)!=1 or len(native_targets)!=1:
                # No fuzzy name match, especially with overloads and aliases.
                continue
            origin=targets[0]; target=native_targets[0]
            graph=graphs[origin['path']]
            source=(root/origin['path']).read_bytes()
            source_lines=source.decode('utf8',errors='replace').splitlines()
            if not 1<=origin['line']<=len(source_lines):continue
            if target['id'] not in {n['id'] for n in graph['nodes']}:
                graph['nodes'].append({**target,'external':True})
            key=(origin['id'],target['id'],path,registration_line)
            if any((e.get('source'),e.get('target'),e.get('registration_path'),e.get('registration_line'))==key
                   for e in graph['edges']):continue
            graph['edges'].append({
                'source':origin['id'],'target':target['id'],
                'source_path':origin['path'],'target_path':path,
                'relation':'jni_register_natives',
                'resolution':'candidate_jni_registration',
                'analyzer':'clang-jni-registration-ast',
                'registration_path':path,'registration_line':registration_line,
                'registration_source_sha256':table_hash,
                'jni_method_descriptor':desc,
                'evidence':{'path':origin['path'],'line':origin['line'],
                            'source_sha256':hashlib.sha256(source).hexdigest(),
                            'snippet':source_lines[origin['line']-1].strip()[:200]}
            })
    if len(paths)>MAX_FILES:
        warnings.append({'reason':'jni_registration_file_limit','limit':MAX_FILES})
    return warnings
