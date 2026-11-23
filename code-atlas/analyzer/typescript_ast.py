"""Optional offline JS/TS symbol parser using the vendored TypeScript compiler.

The analyzed project is read only, never imported or executed. The compiler is
shipped under Apache-2.0; Node.js is an optional runtime. Without Node the
caller must fall back to labeled lexical declarations, never fake AST results.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

ADAPTER = Path(__file__).with_name('typescript_ast_dump.cjs')
COMPILER = Path(__file__).parent / 'vendor' / 'typescript' / 'typescript.js'
MAX_FILE = 2_000_000
MAX_BATCH_BYTES = 16_000_000


def _node_executable():
    # Use a user-configured offline Node binary when PATH does not expose one.
    # No binary is ever taken from the analyzed project's dependency folders.
    configured = os.environ.get('ATLAS_NODE')
    if configured:
        target = Path(configured).expanduser()
        return str(target.resolve()) if target.is_file() else None
    return shutil.which('node') or shutil.which('node.exe')


def analyze_typescript(root: Path, candidates: dict[str, dict], file_graph: dict[str, Any]) -> tuple[dict, list]:
    """Parse self-contained batches; document skipped files rather than hide them.

    Static cross-file symbols are only linked when both modules are present in
    the same batch and an import target is unambiguous. No guesses across batches.
    """
    warnings: list[dict] = []
    paths = sorted(p for p, info in candidates.items() if info.get('language') in {'javascript','typescript'})
    if not paths:
        return {}, warnings
    node = _node_executable()
    if not node or not ADAPTER.is_file() or not COMPILER.is_file():
        warnings.append({'reason':'typescript_ast_unavailable',
            'details':'Node.js (configure ATLAS_NODE) or bundled TypeScript compiler missing; lexical declarations only, no verified calls.'})
        return {}, warnings
    batches: list[list[dict]] = []
    batch: list[dict] = []
    size = 0
    for p in paths:
        file = (root / p).resolve()
        if not file.is_file() or not file.is_relative_to(root) or file.stat().st_size > MAX_FILE:
            warnings.append({'path':p,'reason':'typescript_ast_missing_or_large_file'})
            continue
        try:
            data = file.read_bytes()
            source = data.decode('utf-8-sig')
        except (OSError, UnicodeError):
            warnings.append({'path':p,'reason':'typescript_ast_unreadable_source'})
            continue
        # For stable evidence hashes, hash source bytes as stored on disk.
        entry = {'path':p,'source':source,'hash':hashlib.sha256(data).hexdigest()}
        extra = len(data)
        if batch and size + extra > MAX_BATCH_BYTES:
            batches.append(batch)
            batch=[];size=0
        batch.append(entry);size+=extra
    if batch:
        batches.append(batch)
    result = {}
    for i, entries in enumerate(batches):
        chosen={e['path'] for e in entries}
        edges=[]
        for e in file_graph.get('edges',[]):
            if (e.get('source') not in chosen or e.get('target') not in chosen or
                    e.get('type')!='import' or e.get('confidence_v3') not in {'resolved_static','exact_syntax'}):
                continue
            for ev in e.get('evidence',[]):
                if ev.get('path')==e.get('source') and isinstance(ev.get('line'),int):
                    edges.append({'source':e['source'],'target':e['target'],'line':ev['line'],
                                  'resolution':'resolved_static'})
        payload = json.dumps({'files':entries,'file_edges':edges},ensure_ascii=False)
        try:
            process = subprocess.run([node, str(ADAPTER)], input=payload,text=True,stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,timeout=50,check=False, cwd=ADAPTER.parent)
            if process.returncode!=0:
                raise ValueError(process.stderr[:300] or 'Node parser exited without success')
            data=json.loads(process.stdout)
            if data.get('schema')!='atlas-typescript-ast/0.1':
                raise ValueError('Unexpected compiler AST schema')
            for p, graph in data.get('files',{}).items():
                if p in chosen:
                    if graph.get('parse_errors'):
                        # The TypeScript parser recovers from malformed input.
                        # Recovery AST nodes are not a sound declaration index:
                        # do not return them and never let a legacy lexer
                        # fabricate substitute declarations from this file.
                        warnings.append({'path':p,'reason':'typescript_ast_parse_errors',
                                         'count':len(graph['parse_errors']),
                                         'semantic_nodes_published':False})
                        continue
                    result[p]=graph
        except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            warnings.append({'reason':'typescript_ast_batch_failed','batch':i+1,'files':len(entries),
                             'details':str(exc)[:300]})
    if len(batches)>1:
        warnings.append({'reason':'typescript_ast_split_batches','batches':len(batches),
                         'details':'Mezi oddělenými dávkami nemusí být vyřešeny symbolové reference.'})
    return result, warnings
