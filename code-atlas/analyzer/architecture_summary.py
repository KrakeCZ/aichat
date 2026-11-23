"""Project-neutral, evidence-based structural architecture detection.

No project names, product domains, or filenames from analyzed applications are used
as heuristics. A source tree is not necessarily an MVC/clean/microservice design:
we report *observed* build boundaries, directories, and statically-resolved edges.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import PurePosixPath
import re

# Format signatures are universal syntax, never a particular user's project names.
MANIFEST_SIGNATURES = {
    'package.json': 'Node.js / JavaScript',
    'pyproject.toml': 'Python build',
    'setup.py': 'Python build',
    'Cargo.toml': 'Rust / Cargo',
    'go.mod': 'Go module',
    'go.work': 'Go workspace',
    'build.gradle.kts': 'Gradle',
    'build.gradle': 'Gradle',
    'settings.gradle.kts': 'Gradle workspace',
    'settings.gradle': 'Gradle workspace',
    'pom.xml': 'Maven / JVM',
    'CMakeLists.txt': 'CMake',
    'composer.json': 'PHP / Composer',
    'Gemfile': 'Ruby / Bundler',
    'mix.exs': 'Elixir / Mix',
    'pubspec.yaml': 'Dart / Flutter',
    'Makefile': 'Make',
    'Dockerfile': 'Container build',
}

GENERIC_CONTAINERS = {'src', 'source', 'sources', 'lib', 'libs', 'code'}


def _kind(path: str) -> str | None:
    name = PurePosixPath(path).name
    if name in MANIFEST_SIGNATURES:
        return MANIFEST_SIGNATURES[name]
    if name.lower().endswith(('.sln', '.csproj', '.fsproj', '.vbproj')):
        return '.NET build'
    if name == 'project.pbxproj' and '.xcodeproj/' in path:
        return 'Xcode'
    return None


def detect_manifests(files: dict[str, object]) -> list[dict]:
    return [dict(path=p, directory=str(PurePosixPath(p).parent).replace('.', '', 1)
                 if str(PurePosixPath(p).parent) == '.' else str(PurePosixPath(p).parent),
                 ecosystem=k, basis='observed_manifest_location')
            for p in sorted(files) if (k := _kind(p))]


def _module_assignment(paths: list[str], manifests: list[dict]) -> dict[str, str]:
    """Nearest enclosing manifest wins, otherwise physical root directory."""
    roots = sorted({m['directory'] for m in manifests}, key=lambda r: (-len(PurePosixPath(r).parts), r))
    result: dict[str, str] = {}
    for p in paths:
        nearest = next((r for r in roots if r and p.startswith(r + '/')), None)
        if nearest:
            result[p] = nearest
        elif '' in roots:
            result[p] = '(kořen projektu)'
        else:
            parts = PurePosixPath(p).parts
            result[p] = parts[0] if len(parts) > 1 else '(kořen projektu)'
    return result


def _folder_assignment(paths: list[str], files: dict) -> dict[str, str]:
    root_names = {PurePosixPath(p).parts[0] for p in paths if len(PurePosixPath(p).parts) > 1}
    # More granular buckets for single src-like root, without assuming domain names.
    split = len(root_names) == 1 and next(iter(root_names), '') in GENERIC_CONTAINERS and len(paths) >= 8
    result = {}
    for p in paths:
        parts = PurePosixPath(p).parts
        if len(parts) == 1:
            result[p] = '(kořen projektu)'
        elif split and len(parts) > 2:
            result[p] = '/'.join(parts[:2])
        else:
            result[p] = parts[0]
    if len(set(result.values())) == 1 and len(paths) > 1:
        return {p: 'jazyk: ' + str(files[p].properties.get('language', 'unknown')) for p in paths}
    return result


def _summarize_grouping(files, edges, assignment, max_areas=12):
    paths = sorted(files)
    sizes = Counter(assignment.values())
    popular = {k for k, _ in sorted(sizes.items(), key=lambda item: (-item[1], item[0]))[:max_areas]}
    other = f'Další skupiny ({max(0, len(sizes) - max_areas)})'
    assigned = {p: assignment[p] if assignment[p] in popular else other for p in paths}
    internal, links, degree, types = Counter(), Counter(), Counter(), Counter()
    for e in edges:
        if e.source not in files or e.target not in files:
            continue
        a, b = assigned[e.source], assigned[e.target]
        degree[e.source] += 1
        degree[e.target] += 1
        types[e.type.value if hasattr(e.type, 'value') else str(e.type)] += 1
        if a == b:
            internal[a] += 1
        else:
            links[a, b] += 1
    grouped = defaultdict(list)
    for p, area in assigned.items():
        grouped[area].append(p)
    areas = []
    for name, members in sorted(grouped.items(), key=lambda v: (-len(v[1]), v[0])):
        langs = Counter(str(files[p].properties.get('language') or 'unknown') for p in members)
        highest = sorted(members, key=lambda p: (-degree[p], p))
        areas.append({'id': name, 'label': name, 'files': len(members),
                      'inside_edges': internal[name], 'all_files': sorted(members),
                      'example_files': highest[:5], 'merged': name == other,
                      'languages': [{'name': lang, 'count': count} for lang, count in langs.most_common(5)]})
    conns = [{'source': a, 'target': b, 'count': count} for (a, b), count in
             sorted(links.items(), key=lambda v: (-v[1], v[0]))]
    return {'areas': areas, 'connections': conns,
            'links_inside': sum(internal.values()), 'links_total': sum(links.values()),
            'edge_types': [{'type': k, 'count': v} for k, v in types.most_common()],
            'hotspots': [{'path': p, 'relationships': degree[p]} for p in
                         sorted(paths, key=lambda p: (-degree[p], p))[:12]]}


def _connected_regions(paths: list[str], edges, *, max_communities=15):
    """Weakly connected evidence subgraphs (not inferred application services)."""
    parent = {p: p for p in paths}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for edge in edges:
        if edge.source in parent and edge.target in parent:
            a, b = find(edge.source), find(edge.target)
            if a != b: parent[max(a, b)] = min(a, b)
    grouped = defaultdict(list)
    for p in paths:
        grouped[find(p)].append(p)
    regions = sorted(grouped.values(), key=lambda members: (-len(members), members[0]))
    return [{'label': f'Síť {i+1}', 'file_count': len(r),
             'files': r[:250], 'truncated': len(r) > 250,
             'basis': 'weakly_connected_static_edges'} for i, r in enumerate(regions[:max_communities])]


def summarize_architecture(graph, *, max_areas=12):
    files = {n.id: n for n in graph.nodes.values() if n.type.value == 'file'}
    paths = sorted(files)
    edges = graph.edges
    manifests = detect_manifests(files)
    from_modules = _module_assignment(paths, manifests)
    from_folders = _folder_assignment(paths, files)
    # Prefer explicit nested manifests only if they separate multiple actual areas.
    has_separate_modules = len(set(from_modules.values())) > 1
    grouping = 'adresáře s build manifesty' if has_separate_modules and manifests else 'adresáře / jazyky'
    selected = from_modules if has_separate_modules and manifests else from_folders
    physical = _summarize_grouping(files, edges, from_folders, max_areas)
    boundaries = _summarize_grouping(files, edges, from_modules, max_areas)
    display = _summarize_grouping(files, edges, selected, max_areas)
    languages = Counter(n.properties.get('language', 'unknown') for n in files.values())
    declared = [{'path': m['path'], 'directory': m['directory'] or '(kořen projektu)',
                 'ecosystem': m['ecosystem'], 'basis': 'nalezený soubor manifestu'} for m in manifests]
    return {
        'schema_version': 2,
        'title': 'Zjištěná struktura projektu (statická)',
        'grouping': grouping,
        'note': 'Skupiny odpovídají skutečným adresářům či deklarovaným build hranicím. Hrany jsou jen z nalezených statických důkazů. Architektonický vzor, runtime tok ani úplnost nelze vyvozovat z názvů složek.',
        'files': len(paths),
        'detected_ecosystems': declared,
        'languages': [{'name': k, 'files': v} for k, v in languages.most_common()],
        'views': {'folders': physical, 'build_modules': boundaries},
        'connected_regions': _connected_regions(paths, edges),
        'confidence': {'groups': 'observed_structure', 'links': 'see_graph_edge_evidence',
                       'runtime_architecture': 'not_determined'},
        **display,
    }
