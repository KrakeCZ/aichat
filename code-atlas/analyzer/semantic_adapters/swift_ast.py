"""Swift compiler-backed source AST adapter, with source-location based binding.

Use swiftc -frontend -dump-ast (typecheck + printed AST), never execute code.
Only explicitly compiler-bound direct references are resolved; extension/dynamic
and member calls remain unclaimed. The compiler output is not a stable API;
AST printer changes must fail conservatively. No project-specific assumptions.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from analyzer.symbol_graph import _file_id, _node_id

MAX_FILE_BYTES = 800_000
MAX_FILES = 80
MAX_OUTPUT_BYTES = 14_000_000
DECLS = {
    'func_decl': 'function', 'constructor_decl': 'constructor',
    'struct_decl': 'struct', 'class_decl': 'class', 'enum_decl': 'enum',
    'protocol_decl': 'protocol', 'actor_decl': 'actor',
    'typealias_decl': 'type_alias', 'var_decl': 'variable',
}
CONTAINERS = {'class','struct','enum','protocol','actor'}
DECL_RE = re.compile(r'^\s*\((?:original_init=|processed_init=)?([a-z_]+)\b')
QUOTED_NAME = re.compile(r' "([^"]+)"(?=\s|$)')
SOURCE_RANGE = re.compile(r'range=\[.+?:(\d+):(\d+)\s+-\s+')
SOURCE_LOCATION = re.compile(r'location=.+?:(\d+):(\d+)\b')
BOUND_DECL = re.compile(r'\bdecl="[^"]*?@(.+):(\d+):(\d+)"')


def _name_col(source_line: str, name: str, start_col: int) -> int:
    if not name:
        return start_col
    # Source spans use UTF-8 offsets; exact positions of non-ASCII identifiers
    # are not currently dependable, so leave those AST declarations unbound.
    if not name.isascii():
        return 0
    offsets = [m.start()+1 for m in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', source_line)]
    nearest = next((col for col in offsets if col >= start_col), None)
    return nearest or (offsets[0] if len(offsets) == 1 else 0)


def analyze_swift(root: Path, candidates: dict, _graph: dict) -> tuple[dict, list]:
    paths = sorted(p for p, item in candidates.items() if item.get('language') == 'swift')
    if not paths:
        return {}, []
    compiler = shutil.which('swiftc')
    if not compiler:
        return {}, [{'reason':'swift_compiler_unavailable', 'details':'Swift compiler not installed.'}]
    root = root.resolve()
    chosen = {}; warnings = []
    for path in paths[:MAX_FILES]:
        file = (root / path).resolve()
        if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size > MAX_FILE_BYTES:
            warnings.append({'path':path,'reason':'swift_source_skipped'});continue
        raw = file.read_bytes()
        try:
            lines = raw.decode('utf-8-sig').splitlines()
        except UnicodeDecodeError:
            warnings.append({'path':path,'reason':'swift_invalid_encoding'});continue
        chosen[path] = {'abs':file, 'lines':lines, 'hash':hashlib.sha256(raw).hexdigest()}
    if len(paths) > MAX_FILES:
        warnings.append({'reason':'swift_file_limit','limit':MAX_FILES,'count':len(paths)})
    graph = {}
    calls = []
    declarations_at = defaultdict(list)
    abs_to_path = {str(item['abs']):key for key,item in chosen.items()}
    for path, item in chosen.items():
        # Files in the same directory are the only peers assumed to belong to
        # the same module. Cross-directory module relations are not guessed.
        peers = [other for other in chosen if other != path and Path(other).parent == Path(path).parent]
        cmd = [compiler, '-frontend','-dump-ast','-module-name','AtlasSourceIndex',
               '-primary-file',str(item['abs'])] + [str(chosen[p]['abs']) for p in peers]
        try:
            run = subprocess.run(cmd, cwd=str(root), capture_output=True, timeout=35)
        except (OSError,subprocess.TimeoutExpired) as exc:
            warnings.append({'path':path,'reason':'swift_ast_failed','details':str(exc)[:180]});continue
        if run.returncode or len(run.stdout) > MAX_OUTPUT_BYTES:
            warnings.append({'path':path,'reason':'swift_ast_unavailable_or_diagnostics',
                             'details':run.stderr.decode('utf8','replace')[:220]});continue
        dump = run.stdout.decode('utf8','replace')
        if f'(source_file "{item["abs"]}"' not in dump:
            warnings.append({'path':path,'reason':'swift_ast_unexpected_source_file'});continue
        file_node = {'id':_file_id(path),'path':path,'kind':'file',
                     'name':Path(path).name,'line':1,'end_line':len(item['lines']),
                     'confidence':'verified_file'}
        data = {'nodes':[file_node], 'edges':[], 'parser':'swift_compiler_ast', 'hash':item['hash'],
                'caution':'Swift frontend AST/typechecking. Direct compiler-bound calls only; module isolation, dynamic dispatch and macros may be incomplete.'}
        graph[path] = data
        stack = []
        for line in dump.splitlines():
            m = DECL_RE.match(line)
            if not m:
                continue
            indent = len(line) - len(line.lstrip(' '))
            kind = m.group(1)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            owner = next((n for _,n in reversed(stack) if n.get('kind') in CONTAINERS|{'function','method','constructor'}),file_node)
            if kind in DECLS:
                match = QUOTED_NAME.search(line)
                span = SOURCE_RANGE.search(line)
                if match and span:
                    display = match.group(1)
                    name = display.split('(',1)[0] if '(' in display else display
                    if not name:
                        continue
                    source_line = int(span.group(1)); source_col=int(span.group(2))
                    if not (1 <= source_line <= len(item['lines'])):
                        continue
                    col=_name_col(item['lines'][source_line-1],name,source_col)
                    if not col:
                        continue
                    node_kind = DECLS[kind]
                    if node_kind == 'function' and owner['kind'] in CONTAINERS:
                        node_kind = 'method'
                    qual = owner.get('qualname','')
                    qual = f'{qual}.{display}' if qual else display
                    node = {'id':_node_id(path,qual,source_line,col), 'path':path,
                            'name':name,'qualname':qual,'kind':node_kind,
                            'line':source_line,'column':col,'end_line':source_line,
                            'confidence':'verified_syntax','parent':owner['id']}
                    data['nodes'].append(node)
                    data['edges'].append(_edge(owner,node,path,source_line,chosen,'declares','verified_syntax'))
                    declarations_at[(path,source_line,col)].append(node)
                    stack.append((indent,node))
                    continue
            if kind == 'call_expr' and ' implicit ' not in line:
                loc=SOURCE_LOCATION.search(line)
                if not loc:
                    continue
                call_line=int(loc.group(1));call_col=int(loc.group(2))
                if not (1 <= call_line <= len(item['lines'])):
                    continue
                caller=next((n for _,n in reversed(stack) if n.get('kind') in ('function','method','constructor')),None)
                if caller:
                    calls.append({'origin':path,'owner':caller,'line':call_line,'col':call_col,
                                  'indent':indent, 'target':None})
            elif kind in ('declref_expr','member_ref_expr') and calls:
                # Bind ONLY a direct child of call_expr, i.e. the callee;
                # references in arguments and nested expressions are ignored.
                event=calls[-1]
                if event['origin'] != path or indent != event['indent'] + 2 or event['target'] is not None:
                    continue
                if ' function_ref=single' not in line and ' function_ref=unapplied' not in line:
                    continue
                binding=BOUND_DECL.search(line)
                if not binding:
                    continue
                absolute=str(Path(binding.group(1)).resolve())
                target_path=abs_to_path.get(absolute)
                if target_path:
                    event['target']=(target_path,int(binding.group(2)),int(binding.group(3)))
    seen=set()
    for call in calls:
        path=call['origin']
        if path not in graph or not call.get('target'):
            continue
        candidates=declarations_at.get(call['target'],[])
        possibilities=[n for n in candidates if n['kind'] in ('function','method','constructor')]
        if len(possibilities) != 1:
            continue
        target=possibilities[0]; owner=call['owner']
        key=(owner['id'],target['id'],call['line'])
        if key in seen:
            continue
        seen.add(key)
        resolution='resolved_static' if target['kind']=='function' else 'candidate_dispatch'
        graph[path]['edges'].append(_edge(owner,target,path,call['line'],chosen,'calls',resolution))
    return graph,warnings


def _edge(source:dict,target:dict,evidence_path:str,line:int,files:dict,relation:str,resolution:str)->dict:
    f=files[evidence_path]; lines=f['lines']
    return {'source':source['id'],'target':target['id'],
            'source_path':source['path'],'target_path':target['path'],
            'relation':relation,'resolution':resolution,'analyzer':'swift-compiler-ast',
            'evidence':{'path':evidence_path,'line':line,'source_sha256':f['hash'],
                        'snippet':lines[line-1].strip()[:200] if 0<line<=len(lines) else ''}}
