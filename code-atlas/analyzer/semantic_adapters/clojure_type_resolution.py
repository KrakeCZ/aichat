"""Read-only, platform-scoped Clojure project type-hint evidence.

A source declaration does not prove a loaded JVM class, inferred type or
method overload. In .cljc, ONLY the independently projected :clj syntax may
participate in JVM type-hint candidates; the CLJS projection is never used as
JVM class evidence. Every positive link remains candidate-only.
"""
from __future__ import annotations

from .clojure_forms import _definition
from .clojure_semantics import _arity_body
from .clojure_type_hints import literal_type_tag

MAX_HINTS_PER_FILE = 4096
_TYPE_KINDS = frozenset({'type', 'record', 'interface'})


def _jvm_projection(files, path):
    """Return independently scanned JVM forms/namespace, never shared CLJS.

    A selected reader branch is source evidence only; when a projection cannot
    be constructed, callers must not silently analyze the unprojected .cljc.
    """
    record = files[path]
    if path.endswith('.clj'):
        ns = record['namespace'][0]
        return (record['forms'], ns[0] if ns else None)
    if not path.endswith('.cljc'):
        return (None, None)
    projected = record.get('platform_projections', {}).get('clj', {})
    if projected.get('status') != 'candidate_not_semantically_verified':
        return (None, None)
    ns = projected.get('namespace')
    if not isinstance(ns, str) or not ns:
        return (None, None)
    forms = record.get('_platform_projected_forms', {}).get('clj')
    return (forms, ns) if forms is not None else (None, None)


def _declarations(forms):
    """Source anchors of literal type declarations present in a JVM view."""
    records = set()
    for form in forms:
        definition = _definition(form)
        if definition and definition[1] in _TYPE_KINDS:
            name, kind, part = definition
            records.add((name, kind, part.line, part.col))
    return records


def _jvm_projected_sources(path, graph, forms):
    """Extract literal type hints from already-visible projected functions.

    A projected declaration must match an actual form by name and exact source
    coordinates. Never synthesize an owner or re-use a CLJS-only graph node.
    """
    by_location = {}
    for node in graph['nodes']:
        if node['kind'] not in {'function', 'macro', 'type', 'record', 'interface'}:
            continue
        if node.get('source_platform') not in (None, 'clj'):
            continue
        key = (node['kind'], node['name'], node.get('line'), node.get('column'))
        by_location.setdefault(key, []).append(node)

    seen = set()
    result = []
    for form in forms:
        declaration = _definition(form)
        if declaration is None:
            continue
        name, kind, part = declaration
        matches = by_location.get((kind, name, part.line, part.col), ())
        if len(matches) != 1:
            continue
        owner = matches[0]
        if kind in {'function', 'macro'}:
            arities, issue = _arity_body(form)
            if issue is not None:
                continue
            for params, _body in arities:
                if params.kind != 'vector':
                    continue
                for pattern in params.children:
                    if (tag := literal_type_tag(pattern)) is None:
                        continue
                    key = (owner['id'], tag['line'], tag['column'], tag['source_spelling'])
                    if key not in seen:
                        result.append((owner, tag, 'projected_jvm_function_parameter'))
                        seen.add(key)
        elif kind in {'type', 'record'} and len(form.children) >= 3:
            fields = form.children[2]
            if fields.kind != 'vector':
                continue
            for field in fields.children:
                if (tag := literal_type_tag(field)) is None:
                    continue
                key = (owner['id'], tag['line'], tag['column'], tag['source_spelling'])
                if key not in seen:
                    result.append((owner, tag, 'projected_jvm_datatype_field'))
                    seen.add(key)
    return result


def connect_project_type_hints(graphs, files):
    """Connect explicit JVM hints to unique project source declarations ONLY."""
    type_index = {}
    projection_cache = {}
    for path, graph in sorted(graphs.items()):
        forms, namespace = _jvm_projection(files, path)
        projection_cache[path] = (forms, namespace)
        if forms is None or namespace is None:
            continue
        declared = _declarations(forms)
        for node in graph['nodes']:
            if node['kind'] not in _TYPE_KINDS:
                continue
            if node.get('source_platform') not in (None, 'clj'):
                continue
            key = (node['name'], node['kind'], node.get('line'), node.get('column'))
            if key in declared:
                type_index.setdefault((namespace, node['name']), []).append((path, node))

    def hint_target(namespace, spelling):
        if namespace is None:
            return None, 'namespace_not_proven'
        if '/' in spelling or spelling.startswith(':') or not spelling:
            return None, 'non_literal_jvm_generated_class_spelling'
        if '.' in spelling:
            target_ns, simple = spelling.rsplit('.', 1)
        else:
            target_ns, simple = namespace, spelling
        matched = type_index.get((target_ns, simple), [])
        if len(matched) != 1:
            return None, 'source_type_missing_or_ambiguous'
        return matched[0], None

    for path, graph in sorted(graphs.items()):
        forms, namespace = projection_cache[path]
        observations = []
        if forms is None or (namespace is None and path.endswith('.cljc')):
            # Preserve the prior serialized contract for non-JVM sources.
            # A .clj WITHOUT `ns` is still scanned below so its explicit
            # tags remain visible as unresolved evidence rather than vanish.
            graph['clojure_project_type_hint_candidates'] = {
                'status': 'not_a_single_jvm_source_projection', 'hints': []}
            continue
        sources = []
        if path.endswith('.clj'):
            for node in graph['nodes']:
                if node.get('literal_type_hint'):
                    sources.append((node, node['literal_type_hint'], 'field_or_member'))
            semantic = graph.get('clojure_semantic_analysis', {})
            for qual, data in semantic.items():
                owner = next((x for x in graph['nodes']
                    if x.get('qualname') == qual and x['kind'] == 'function'), None)
                if owner is None:
                    continue
                for arity in data.get('arities', []):
                    for record in arity.get('literal_parameter_type_hints', []):
                        sources.append((owner, record['type_tag'], 'function_parameter'))
        else:
            sources.extend(_jvm_projected_sources(path, graph, forms))
        limited = 0
        for owner, tag, kind in sources:
            if len(observations) >= MAX_HINTS_PER_FILE:
                limited += 1
                continue
            matching, issue = hint_target(namespace, tag['source_spelling'])
            record = {
                'source_symbol': owner['id'],
                'spelling': tag['source_spelling'],
                'source': {'path': path, 'line': tag['line'], 'column': tag['column'],
                           'end_line': tag['end_line'], 'end_column': tag['end_column'],
                           'source_sha256': files[path]['sha']},
                'type_hint_kind': kind,
                'source_platform': 'clj',
                'compiler_type_binding_confirmed': False,
                'runtime_type_confirmed': False}
            if matching is not None:
                target_path, target = matching
                record.update(candidate_source_type=target['id'],
                              candidate_source_path=target_path,
                              status='unique_project_source_datatype_candidate')
                graph['edges'].append({
                    'source': owner['id'], 'target': target['id'],
                    'source_path': path, 'target_path': target_path,
                    'relation': 'clojure_project_type_hint_candidate',
                    'resolution': 'unique_source_deftype_defrecord_or_definterface_candidate',
                    'analyzer': 'clojure-static-project-type-hints',
                    'source_platform': 'clj', 'target_platform': 'clj',
                    'runtime_type_confirmed': False,
                    'compiler_type_binding_confirmed': False,
                    'evidence': record['source']})
            else:
                record.update(status='unresolved', reason=issue)
            observations.append(record)
        graph['clojure_project_type_hint_candidates'] = {
            'status': 'jvm_source_only_type_hint_candidates',
            'hints': observations, 'omitted_due_to_limit': limited,
            'compiler_type_binding_confirmed': False}
