"""Conservative source-to-native JNI ABI linker.

Matches native declarations to static C exports by the official JNI short/long
naming convention. Overloaded methods require the *exact* JVM parameter
signature from compiler output. Name matches are candidates, never proof of
runtime loading, registration, process architecture or library availability.
"""
from __future__ import annotations

from collections import defaultdict


def _jni_escape(value: str) -> str:
    out = []
    for c in value:
        if c.isascii() and c.isalnum():
            out.append(c)
        elif c == '_':
            out.append('_1')
        elif c == ';':
            out.append('_2')
        elif c == '[':
            out.append('_3')
        elif c == '/':
            out.append('_')
        else:
            out.append('_0%04x' % ord(c))
    return ''.join(out)


def _native_identity(node: dict) -> tuple[str, str, str] | None:
    if node.get('jvm_owner'):
        return node['jvm_owner'], node['name'], node.get('jvm_descriptor', '')
    # Fallback for syntax-only source files: only simple class-level methods.
    qual = node.get('qualname', '')
    parts = qual.split('.')
    if len(parts) != 2:
        return None
    package = node.get('package_name', '')
    owner = (package.replace('.', '/') + '/' if package else '') + parts[0]
    return owner, parts[1], ''


def connect_native_jni(graphs: dict) -> list[dict]:
    exports = defaultdict(list)
    for graph in graphs.values():
        if graph.get('parser') != 'clang_ast':
            continue
        for node in graph['nodes']:
            if node.get('external') or node.get('kind') != 'function':
                continue
            name = node.get('name', '')
            if name.startswith('Java_'):
                exports[name].append(node)

    natives = defaultdict(list)
    for graph in graphs.values():
        if graph.get('parser') not in ('javac_ast', 'kotlin_psi'):
            continue
        for node in graph['nodes']:
            if node.get('external') or not (node.get('native_declaration') or node.get('external_declaration')):
                continue
            identity = _native_identity(node)
            if identity:
                owner, method, descriptor = identity
                if method and method != 'constructor':
                    short = 'Java_' + _jni_escape(owner) + '_' + _jni_escape(method)
                    natives[short].append((node, descriptor))

    warnings = []
    for short, declarations in natives.items():
        for declaration, desc in declarations:
            matches = []
            matched_by_signature = False
            if desc and desc.startswith('(') and ')' in desc:
                args = desc[1:desc.index(')')]
                matches = exports.get(short + '__' + _jni_escape(args), [])
                matched_by_signature = bool(matches)
            if not matches and len(declarations) == 1:
                matches = exports.get(short, [])
            if len(matches) != 1:
                if len(matches) > 1:
                    warnings.append({'reason':'jni_ambiguous_export','path':declaration['path']})
                continue
            export = matches[0]
            graph = graphs.get(declaration['path'])
            if not graph:
                continue
            if export['id'] not in {node['id'] for node in graph['nodes']}:
                graph['nodes'].append({**export, 'external': True})
            graph['edges'].append({
                'source': declaration['id'], 'target': export['id'],
                'source_path': declaration['path'], 'target_path': export['path'],
                'relation': 'jni_name_match',
                'resolution': 'candidate_jni_abi_signature' if matched_by_signature else 'candidate_jni_abi_name',
                'analyzer': 'jni-abi-source-matcher',
                'evidence': {
                    'path': declaration['path'], 'line': declaration['line'],
                    'source_sha256': graph['hash'],
                    'snippet': 'Native method declaration; matching exported JNI symbol (runtime not proven).'
                }
            })
    return warnings
