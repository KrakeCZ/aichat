"""Optional Clang AST adapter for C/C++ sources and their local headers.

Uses Clang JSON AST without executing user code or building binaries.  Exposes
verified declarations and direct call-site references, not link-time/runtimes.
Compilation databases, macros, templates and virtual dispatch need separate
resolution passes; uncertainty is explicitly labeled.
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
from typing import Any

from analyzer.symbol_graph import _file_id, _node_id

MAX_FILE = 1_000_000
MAX_OUTPUT = 32_000_000
MAX_UNITS = 100
DECL_KINDS = {
    'FunctionDecl':'function', 'CXXMethodDecl':'method', 'CXXConstructorDecl':'constructor',
    'CXXDestructorDecl':'destructor', 'CXXConversionDecl':'method',
    'RecordDecl':'record', 'CXXRecordDecl':'class', 'EnumDecl':'enum',
    'FieldDecl':'field', 'VarDecl':'variable', 'ParmVarDecl':'parameter',
    'ObjCInterfaceDecl':'class', 'ObjCImplementationDecl':'class',
    'ObjCProtocolDecl':'protocol', 'ObjCCategoryDecl':'category',
    'ObjCMethodDecl':'method', 'ObjCPropertyDecl':'property', 'ObjCIvarDecl':'field',
}
CALL_KINDS = {'CallExpr','CXXMemberCallExpr','CXXOperatorCallExpr'}


def _clang_executable():
    configured = os.environ.get('ATLAS_CLANG')
    if configured:
        target = Path(configured).expanduser()
        return str(target.resolve()) if target.is_file() else None
    return shutil.which('clang++') or shutil.which('clang')


def analyze_clang(root: Path, candidates: dict, _file_graph: dict, *, languages=('c','cpp')) -> tuple[dict, list]:
    units = sorted(p for p,v in candidates.items() if v.get('language') in languages)
    if not units:
        return {}, []
    warnings=[]
    compiler = _clang_executable()
    if not compiler:
        return {}, [{'reason':'clang_compiler_unavailable', 'details':'Install a local Clang executable or configure ATLAS_CLANG. No analyzed project code is executed.'}]
    root = root.resolve()
    chosen = {}
    for p, info in candidates.items():
        if info.get('language') not in (*languages, *(['header'] if 'c' in languages or 'cpp' in languages else [])):
            continue
        f=(root/p).resolve()
        if not f.is_relative_to(root) or not f.is_file() or f.stat().st_size>MAX_FILE:
            continue
        raw=f.read_bytes()
        chosen[p]={'raw':raw, 'lines':raw.decode('utf8',errors='replace').splitlines(),
                   'hash':hashlib.sha256(raw).hexdigest(), 'offsets':None}
    for p,data in chosen.items():
        data['offsets']=[0]+[i+1 for i,b in enumerate(data['raw']) if b==10]
    all_nodes = defaultdict(dict)
    all_calls = []
    observed=set()
    for path in units[:MAX_UNITS]:
        if path not in chosen:
            warnings.append({'path':path,'reason':'clang_file_skipped'});continue
        suffix=Path(path).suffix.lower()
        lang={'c':'c', 'm':'objective-c','mm':'objective-c++'}.get(suffix.lstrip('.'),'c++')
        standard='c11' if lang=='c' else 'c++17'
        std_arg=['-std='+standard] if lang in ('c','c++','objective-c++') else []
        from .jni_registration import _jni_headers
        jni_include = _jni_headers() if b'jni.h' in chosen[path]['raw'] else []
        cmd=[compiler,'-x',lang,*std_arg,*jni_include,'-fsyntax-only','-Xclang','-ast-dump=json',
             '-Wno-everything','-I',str(root),str(root/path)]
        try:
            process=subprocess.run(cmd,text=True,capture_output=True,timeout=22,cwd=str(root))
            if len(process.stdout)>MAX_OUTPUT:
                warnings.append({'path':path,'reason':'clang_ast_output_too_large'});continue
            tree=json.loads(process.stdout)
        except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            warnings.append({'path':path,'reason':'clang_ast_unavailable_or_failed','details':str(exc)[:180]});continue
        if process.returncode:
            warnings.append({'path':path,'reason':'clang_ast_diagnostics',
                             'details':process.stderr.strip()[:220]})
        # Across a translation unit, Clang omits location.file from repeated
        # declarations; location offsets are relative to the last file switch.
        previous=path
        refs={}
        nodes=[]
        calls=[]
        def loc(n, assumed):
            raw=n.get('loc') or n.get('range',{}).get('begin') or {}
            name=raw.get('file')
            if not name:
                name=(n.get('range',{}).get('begin') or {}).get('file')
            file=assumed
            if name:
                abs_path=Path(name).resolve()
                file=abs_path.relative_to(root).as_posix() if abs_path.is_relative_to(root) else ''
            if file not in chosen:
                return file,0,0
            try:
                offset=int(raw.get('offset',-1))
                if offset>=0:
                    ln=bisect.bisect_right(chosen[file]['offsets'],offset)
                    col=offset-chosen[file]['offsets'][ln-1]+1
                    return file,ln,col
                return file,int(raw.get('line',0)),int(raw.get('col',0))
            except (TypeError,ValueError):
                return file,0,0
        def walk(n, file, owner=None, qualprefix=''):
            if not isinstance(n,dict):return
            nonlocal previous
            kind=n.get('kind','')
            # Top-level declarations may switch from included header to TU.
            actual,ln,col=loc(n,file)
            current=actual if actual in chosen else file
            # Reject declarations from included system headers when Clang
            # omits `file` on subsequent AST locations. Their offsets belong
            # to another file and must not be attributed to this source.
            src_offset=(n.get('loc') or {}).get('offset',-1)
            valid_name=(isinstance(src_offset,int) and src_offset>=0
                and actual in chosen and chosen[actual]['raw'][src_offset:src_offset+len(n.get('name','').encode('utf8'))]==n.get('name','').encode('utf8'))
            if kind == 'ObjCMethodDecl' and isinstance(src_offset,int) and src_offset >= 0 and actual in chosen:
                # Clang loc.offset for Objective-C methods points to '-' or '+',
                # NOT to the selector name. Check the actual method signature
                # at that AST location; keep the strict byte check for others.
                fragment=chosen[actual]['raw'][src_offset:src_offset+512].decode('utf8',errors='replace')
                selector=n.get('name','').split(':',1)[0]
                valid_name=bool(selector and re.match(r'[-+]\s*\([^)]*\)\s*'+re.escape(selector)+r'\b',fragment))
            if (kind in DECL_KINDS and not n.get('isImplicit') and n.get('name')
                    and ln>0 and col>0 and actual in chosen and valid_name):
                name=n['name']; qual=qualprefix+'.'+name if qualprefix else name
                # Keep overloads disambiguated by location, not project names.
                node={'id':_node_id(actual,qual,ln,col),'path':actual,'name':name,
                      'qualname':qual,'kind':DECL_KINDS[kind],'line':ln,'column':col,
                      'confidence':'verified_syntax',
                      'end_line':ln,
                      'has_body':kind in {'FunctionDecl','CXXMethodDecl','CXXConstructorDecl','ObjCMethodDecl'} and any(child.get('kind')=='CompoundStmt' for child in n.get('inner',[]))}
                if owner and owner['path']==actual:
                    node['parent']=owner['id']
                elif owner is None or owner['path']!=actual:
                    node['parent']=_file_id(actual)
                if n.get('id'):
                    refs[n['id']]=node
                if n.get('previousDecl') and n['previousDecl'] in refs:
                    refs[n['previousDecl']]=refs[n['previousDecl']]
                nodes.append(node)
                new_owner=node if kind in {'CXXRecordDecl','RecordDecl','FunctionDecl','CXXMethodDecl','CXXConstructorDecl','CXXDestructorDecl','CXXConversionDecl','ObjCImplementationDecl','ObjCInterfaceDecl','ObjCProtocolDecl','ObjCMethodDecl'} else owner
                new_prefix=qual if kind in {'CXXRecordDecl','RecordDecl','EnumDecl','ObjCImplementationDecl','ObjCInterfaceDecl','ObjCProtocolDecl'} else qualprefix
            else:
                new_owner=owner;new_prefix=qualprefix
            if kind in CALL_KINDS and owner and actual in chosen and ln>0:
                calls.append((owner,actual,ln,n,kind))
            for child in n.get('inner',[]):
                if not isinstance(child,dict):continue
                # Clang commonly omits source locations for children, but the
                # nearest enclosing declaration identifies their source file.
                child_file,_,_=loc(child,current)
                walk(child,child_file if child_file in chosen else current,new_owner,new_prefix)
        # Top-level translation unit nodes can contain included declarations.
        for top in tree.get('inner',[]):
            if not isinstance(top,dict):continue
            fp,_,_=loc(top,previous)
            if fp in chosen:previous=fp
            walk(top,previous)
        for node in nodes:
            all_nodes[node['path']][node['id']]=node
        for owner,fp,ln,call,kind in calls:
            def references(node):
                if not isinstance(node,dict):return []
                arr=[]
                if node.get('kind') in {'DeclRefExpr','MemberExpr'}:
                    target=node.get('referencedDecl') or {}
                    if isinstance(target,dict) and target.get('id'):
                        arr.append(target['id'])
                    if isinstance(node.get('referencedMemberDecl'),str):
                        arr.append(node['referencedMemberDecl'])
                for child in node.get('inner',[]):arr.extend(references(child))
                return arr
            hits=[refs.get(r) for r in references(call)]
            funcs=[item for item in hits if item and item['kind'] in {'function','method','constructor','destructor'}]
            unique={x['id']:x for x in funcs}
            if len(unique)!=1:continue
            target=next(iter(unique.values()))
            # A function declaration without a body may resolve to a separate
            # linked implementation. Virtual and C++ member calls can dispatch.
            resolved=kind=='CallExpr' and target['kind']=='function' and target.get('has_body',False)
            resolution='resolved_static' if resolved else 'candidate_dispatch' if kind!='CallExpr' else 'resolved_declaration'
            all_calls.append((owner,target,fp,ln,resolution))
    results={}
    for path, symbols in all_nodes.items():
        data=chosen[path]
        first={'id':_file_id(path),'path':path,'name':path.rsplit('/',1)[-1],
               'kind':'file','line':1,'end_line':len(data['lines']), 'confidence':'verified_file'}
        nodes=[first]+list(symbols.values())
        edges=[]
        for item in symbols.values():
            # The source file is always a verifiable declaration owner, even
            # if a nested scope additionally owns this symbol.
            parent=next((p for p in nodes if p['id']==item.get('parent')), first)
            edges.append(_edge(parent,item,path,item['line'],'declares','verified_syntax',chosen))
        results[path]={'nodes':nodes,'edges':edges,'parser':'clang_objc_ast' if path.endswith(('.m','.mm')) else 'clang_ast','hash':data['hash'],
                       'caution':'Clang source AST, not linked binary; macros, templates, virtual calls and include paths may be incomplete.'}
    for owner,target,fp,ln,res in all_calls:
        if fp not in results or target['path'] not in chosen:
            continue
        if owner['id'] not in {x['id'] for x in results[fp]['nodes']}:
            continue
        nodes=results[fp]['nodes']
        if target['id'] not in {x['id'] for x in nodes}:
            nodes.append({**target,'external':True})
        results[fp]['edges'].append(_edge(owner,target,fp,ln,'calls',res,chosen))
    if len(units)>MAX_UNITS:
        warnings.append({'reason':'clang_ast_translation_unit_limit','limit':MAX_UNITS})
    return results,warnings


def _edge(a,b,path,line,relation,resolution,chosen):
    raw=chosen[path];lines=raw['lines']
    return {'source':a['id'],'target':b['id'],
            'source_path':a['path'],'target_path':b['path'],
            'relation':relation,'resolution':resolution,
            'analyzer':'clang-source-ast',
            'evidence':{'path':path,'line':line,'source_sha256':raw['hash'],
                        'snippet':lines[line-1].strip()[:200] if 0<line<=len(lines) else ''}}
