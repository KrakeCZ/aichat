"""JDK javac AST and conservative symbol references, not a Java code executor.

This adapter is discovered by language; no assumptions about project structure,
package names or framework. Missing build dependencies are diagnosed, not faked.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from analyzer.symbol_graph import _node_id, _file_id

BRIDGE = Path(__file__).with_name('AtlasJavaAstBridge.java')
MAX_FILE = 2_000_000
MAX_FILES = 1000


def _unb64(s: str) -> str:
    return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4)).decode('utf-8')


def analyze_java(root: Path, candidates: dict, _file_graph: dict) -> tuple[dict, list]:
    paths = sorted(p for p, i in candidates.items() if i.get('language') == 'java')
    if not paths:
        return {}, []
    warnings = []
    if not shutil.which('java') or not shutil.which('javac'):
        return {}, [{'reason': 'java_jdk_unavailable', 'details': 'Java Compiler API requires an installed JDK.'}]
    chosen = []
    hashes = {}
    source_lines = {}
    root = root.resolve()
    for p in paths:
        f = (root / p).resolve()
        if not f.is_relative_to(root) or not f.is_file() or f.stat().st_size > MAX_FILE:
            warnings.append({'path': p, 'reason': 'java_ast_file_skipped'}); continue
        if len(chosen) >= MAX_FILES:
            warnings.append({'reason': 'java_ast_file_limit', 'limit': MAX_FILES}); break
        raw = f.read_bytes()
        hashes[p] = hashlib.sha256(raw).hexdigest()
        source_lines[p] = raw.decode('utf-8-sig', errors='replace').splitlines()
        chosen.append(p)
    if not chosen:
        return {}, warnings
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-javac-') as temp:
            temp = Path(temp)
            manifest = temp / 'sources.list'
            manifest.write_text('\n'.join(chosen) + '\n', encoding='utf8')
            proc = subprocess.run(['javac', '-encoding', 'UTF-8', '-d', str(temp), str(BRIDGE)],
                                  text=True, capture_output=True, timeout=40)
            if proc.returncode:
                raise RuntimeError(proc.stderr[:500])
            proc = subprocess.run(['java', '-Xmx512m', '-cp', str(temp), 'AtlasJavaAstBridge',
                                   str(root), str(manifest)], text=True, capture_output=True,
                                  timeout=55, cwd=str(temp))
            if proc.returncode:
                raise RuntimeError(proc.stderr[:500])
            stdout = proc.stdout
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {}, warnings + [{'reason': 'java_ast_bridge_failed', 'details': str(exc)[:350]}]
    output = {}
    index = {}
    pending = []
    packages = {}
    native_records = []
    for path in chosen:
        output[path] = {
            'nodes': [{'id': _file_id(path), 'path': path, 'name': path.rsplit('/', 1)[-1],
                       'kind': 'file', 'line': 1, 'end_line': len(source_lines[path]),
                       'confidence': 'verified_file'}],
            'edges': [], 'parser': 'javac_ast', 'hash': hashes[path],
            'caution': 'JDK AST parsed without executing project code. Virtual dispatch remains a candidate; classpath may be incomplete.'}
    for record in stdout.splitlines():
        fields = record.split('\t')
        if not fields:
            continue
        kind = fields[0]
        try:
            if kind == 'P' and len(fields) == 3:
                packages[_unb64(fields[1])] = _unb64(fields[2])
            elif kind == 'N' and len(fields) in (9, 10):
                _, fp, k, nm, q, parent, line, col, endline = fields[:9]
                path, k, nm, q, parent = map(_unb64, (fp, k, nm, q, parent))
                if path not in output:
                    continue
                line, col, endline = int(line), int(col), int(endline)
                if not 1 <= line <= len(source_lines[path]):
                    continue
                is_native = len(fields) == 10 and _unb64(fields[9]) == 'true'
                node = {'id': _node_id(path, q, line, col), 'path': path, 'kind': k,
                        'name': nm, 'qualname': q, 'line': line, 'column': col,
                        'end_line': endline, 'confidence': 'verified_syntax',
                        'native_declaration': is_native,
                        '_parent_qual': parent}
                key = (path, line, col)
                # No spurious duplicates on repeated compiler declarations.
                if key not in index:
                    index[key] = node
                    output[path]['nodes'].append(node)
            elif kind == 'J' and len(fields) == 7:
                native_records.append((_unb64(fields[1]), int(fields[2]), int(fields[3]),
                                       _unb64(fields[4]),_unb64(fields[5]),_unb64(fields[6])))
            elif kind == 'C' and len(fields) == 8:
                _, fp, origin, call_line, tp, target_line, target_col, resolution = fields
                pending.append((_unb64(fp), _unb64(origin), int(call_line),
                                _unb64(tp), int(target_line), int(target_col), resolution))
            elif kind == 'E':
                warnings.append({'reason': 'java_ast_compiler_diagnostic',
                                 'path': _unb64(fields[1])[:180],
                                 'line': int(fields[2]), 'details': _unb64(fields[3])[:170]})
            elif kind == 'W':
                warnings.append({'reason': 'java_ast_partial', 'details': _unb64(fields[2])[:170]})
        except (ValueError, UnicodeError, IndexError, KeyError):
            warnings.append({'reason': 'java_ast_invalid_record'})
    for path, ln, col, owner, name, descriptor in native_records:
        method = index.get((path, ln, col))
        if method and method.get('native_declaration') and method['name'] == name:
            method['jvm_owner'] = owner
            method['jvm_descriptor'] = descriptor
    for path, graph in output.items():
        for node in graph['nodes']:
            node['package_name'] = packages.get(path, '')
    by_qual = defaultdict(list)
    for data in output.values():
        for n in data['nodes']:
            by_qual[(n['path'], n.get('qualname'))].append(n)
    for path, data in output.items():
        for node in data['nodes'][1:]:
            owner = node.pop('_parent_qual', '')
            options = by_qual.get((path, owner), []) if owner else [data['nodes'][0]]
            if len(options) != 1:
                continue
            parent = options[0]
            node['parent'] = parent['id']
            data['edges'].append(_edge(parent, node, 'declares', 'verified_syntax',
                                       node['line'], path, hashes, source_lines))
    ids_by_file = {p: {n['id'] for n in d['nodes']} for p, d in output.items()}
    seen = set()
    for path, qual, line, target_path, target_line, target_col, res in pending:
        if path not in output or target_path not in output:
            continue
        callers = [n for n in by_qual[(path, qual)] if n['kind'] in ('method', 'constructor')]
        target = index.get((target_path, target_line, target_col))
        if len(callers) != 1 or not target or target['kind'] not in ('method', 'constructor'):
            continue
        key = (path, qual, line, target['id'])
        if key in seen:
            continue
        seen.add(key)
        if target['id'] not in ids_by_file[path]:
            output[path]['nodes'].append({**target, 'external': True})
            ids_by_file[path].add(target['id'])
        output[path]['edges'].append(_edge(callers[0], target, 'calls', res, line,
                                            path, hashes, source_lines))
    return output, warnings


def _edge(a, b, relation, resolution, line, evidence_path, hashes, source_lines):
    lines = source_lines[evidence_path]
    return {'source': a['id'], 'target': b['id'],
            'source_path': a['path'], 'target_path': b['path'],
            'relation': relation, 'resolution': resolution,
            'analyzer': 'javac-compiler-ast',
            'evidence': {'path': evidence_path, 'line': line,
                         'source_sha256': hashes[evidence_path],
                         'snippet': lines[line-1].strip()[:200] if 0 < line <= len(lines) else ''}}
