"""Conservative Android source-set and variant context inferred from literal Gradle DSL.

No Gradle execution; this is NOT an Android Gradle Plugin model.  Variants are
possible declared configurations, not resolved build outputs.  Ambiguous DSL
results in unknown membership, never invented edges or assertions of certainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import posixpath
import re
import tomllib
from pathlib import Path
from typing import Any

from .build_context import _lines, _mask_quoted

# Declared Android plugin id is a language/framework fact, never a project name.
_ANDROID_PLUGIN = re.compile(r'\bcom\.android\.(?:application|library|dynamic-feature|test)\b')
_CREATE = re.compile(r'\b(?:create|maybeCreate|getByName|named)\s*\(\s*["\']([\w.-]+)["\']\s*\)\s*$')
_PLAIN = re.compile(r'([A-Za-z_][\w.-]*)\s*$')
_DIMENSION_VALUE = re.compile(r'\bdimension\s*(?:=|\s)\s*["\']([\w.-]+)["\']')
_DIMENSIONS = re.compile(r'\bflavorDimensions\s*(?:\+?=|\(|\s)')
_LIST_LITERAL = re.compile(r'["\']([\w.-]+)["\']')
_PATH_LITERAL = re.compile(r'["\']([^"\']+)["\']')
_PLUGIN_LITERAL = re.compile(r'\b(?:id\s*(?:\(\s*)?|apply\s*(?:\(\s*)?plugin\s*[:=]\s*)["\'](com\.android\.[\w.-]+)["\']')
_SOURCE_DIR = re.compile(r'\b(?:java|kotlin|res|assets|aidl)\s*\.\s*(?:srcDirs|srcDir|setSrcDirs)\s*(?:\(|\s)')
_SRC_EXT = ('.kt', '.kts', '.java', '.xml', '.aidl', '.cpp', '.cc', '.c', '.h', '.hpp', '.png', '.jpg', '.webp', '.svg')


@dataclass
class Block:
    label: str
    start: int
    end: int
    parent: int | None
    line: int


def _blocks(clean: str) -> list[Block]:
    """Resolve DSL block ancestry without confusing quoted braces with scopes."""
    masked = _mask_quoted(clean)
    blocks: list[Block] = []
    stack: list[int] = []
    segment_start = 0
    for offset, c in enumerate(masked):
        if c == '{':
            before = clean[segment_start:offset].strip()
            m = _CREATE.search(before) or _PLAIN.search(before)
            label = m.group(1) if m else ''
            idx = len(blocks)
            blocks.append(Block(label, offset, len(clean), stack[-1] if stack else None,
                                clean.count('\n', 0, offset) + 1))
            stack.append(idx)
            segment_start = offset + 1
        elif c == '}':
            if stack:
                blocks[stack.pop()].end = offset
            segment_start = offset + 1
    return blocks


def _ancestors(blocks: list[Block], index: int) -> list[str]:
    parts = []
    while index is not None:
        parts.append(blocks[index].label)
        index = blocks[index].parent
    return list(reversed(parts))


def _contents(clean: str, block: Block) -> str:
    return clean[block.start + 1:block.end]


def _literal_dimensions(clean: str, android: Block) -> list[str]:
    """Read literal dimension names without executing DSL expressions."""
    found = []
    for line in _contents(clean, android).splitlines():
        m = _DIMENSIONS.search(line)
        if not m:
            continue
        # Don't interpret property references or unrelated strings as declarations.
        tail = line[m.end():].strip()
        if tail.startswith('listOf(') and ')' in tail:
            tail = tail[len('listOf('):tail.index(')')]
        elif tail.startswith('(') and ')' in tail:
            tail = tail[1:tail.index(')')]
        # Only a comma-separated sequence of literal dimensions is acceptable.
        # e.g. providers.gradleProperty("tier") is dynamic, not "tier".
        if not re.fullmatch(r"\s*['\"][\w.-]+['\"](?:\s*,\s*['\"][\w.-]+['\"])*\s*", tail):
            continue
        vals = _LIST_LITERAL.findall(tail)
        for v in vals:
            if v not in found:
                found.append(v)
    return found


def _source_roots(clean: str, blocks: list[Block], android_index: int, module_dir: str) -> list[dict]:
    """Recognize only explicit literal source directories in Android sourceSets."""
    result = []
    for idx, block in enumerate(blocks):
        chain = _ancestors(blocks, idx)
        if len(chain) < 3 or chain[-3] != 'android' or chain[-2] != 'sourceSets' or idx == android_index:
            continue
        source_set = block.label
        if not source_set or source_set in ('sourceSets', 'android'):
            continue
        content = _contents(clean, block)
        base_line = clean.count('\n', 0, block.start) + 1
        for i, line in enumerate(content.splitlines(), 0):
            m = _SOURCE_DIR.search(line)
            if not m:
                continue
            # Explicitly limited to literal directory arguments. Do not resolve
            # provider APIs, generated directories or interpolated paths.
            for directory in _PATH_LITERAL.findall(line[m.end():]):
                if '$' in directory or directory.startswith(('/', '\\')) or '..' in directory.replace('\\','/').split('/'):
                    continue
                normalized = posixpath.normpath(posixpath.join(module_dir, directory.replace('\\','/')))
                if normalized in ('.','') or normalized.startswith('../'):
                    continue
                result.append({'source_set':source_set, 'root':normalized, 'line':base_line + i,
                               'status':'declared_literal_not_evaluated'})
    return result


def _variant_names(dimensions: list[str], flavors: dict[str, str], types: list[str], limit: int = 96):
    if flavors:
        if not dimensions or any(v not in dimensions for v in flavors.values()):
            return [], 'flavor_dimensions_unresolved'
        grouped = [[name for name, dim in flavors.items() if dim == d] for d in dimensions]
        if any(not choices for choices in grouped):
            return [], 'dimension_has_no_literal_flavor'
    else:
        grouped = []
    count = len(types)
    for group in grouped:
        count *= len(group)
    if count > limit:
        return [], 'too_many_variants_for_static_enumeration'
    combos = product(*grouped) if grouped else [()]
    variants = []
    for combination in combos:
        prefix = combination[0] if combination else ''
        prefix += ''.join(x[:1].upper()+x[1:] for x in combination[1:])
        for build_type in types:
            variant = prefix + (build_type[:1].upper() + build_type[1:] if prefix else build_type)
            variants.append({'name':variant, 'build_type':build_type,
                             'flavors':list(combination), 'source_sets':['main',build_type,*combination,
                             *([prefix] if prefix and prefix not in combination else []), variant]})
    return variants, None


def _plugin_aliases(root: Path) -> set[str]:
    """Recognize standard Gradle version-catalog plugin aliases from actual TOML IDs."""
    aliases = set()
    path = root / 'gradle' / 'libs.versions.toml'
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 1_000_000:
        return aliases
    try:
        catalog = tomllib.loads(path.read_text(encoding='utf8'))
        for name, spec in catalog.get('plugins', {}).items():
            plugin_id = spec.get('id', '') if isinstance(spec, dict) else spec if isinstance(spec, str) else ''
            if _ANDROID_PLUGIN.fullmatch(plugin_id):
                aliases.add('libs.plugins.' + name.replace('-', '.').replace('_', '.'))
    except (OSError, UnicodeError, ValueError, TypeError):
        pass
    return aliases


def _module_model(build_path: str, text: str, aliases: set[str]) -> dict | None:
    codes = [code for _,code,_ in _lines(text)]
    clean = '\n'.join(codes)
    if not any(_ANDROID_PLUGIN.fullmatch(m.group(1)) for m in _PLUGIN_LITERAL.finditer(clean)) and not any(
        re.search(r'\balias\s*\(\s*'+re.escape(alias)+r'\s*\)', clean)
        for alias in aliases
    ):
        return None
    blocks = _blocks(clean)
    android_indices = [i for i,b in enumerate(blocks) if b.label == 'android' and b.parent is None]
    if len(android_indices) != 1:
        return {'build_file':build_path, 'status':'ambiguous_android_configuration', 'variants':[],
                'source_roots':[], 'flavors':{}, 'build_types':[], 'dimensions':[]}
    ai = android_indices[0]
    dimensions = _literal_dimensions(clean, blocks[ai])
    build_types = ['debug','release']  # Android default build types; only when Android plugin is literal.
    flavors: dict[str, str] = {}
    problems = []
    for i, b in enumerate(blocks):
        chain = _ancestors(blocks, i)
        if len(chain) == 3 and chain[0] == 'android':
            if chain[1] == 'buildTypes' and b.label not in build_types and b.label:
                build_types.append(b.label)
            if chain[1] == 'productFlavors' and b.label:
                dim = _DIMENSION_VALUE.search(_contents(clean, b))
                if dim:
                    flavors[b.label] = dim.group(1)
                elif len(dimensions) == 1:
                    # Android requires a dimension; never silently default it.
                    problems.append('flavor_dimension_not_explicit:' + b.label)
                else:
                    problems.append('flavor_dimension_unknown:' + b.label)
    variants, reason = _variant_names(dimensions, flavors, build_types)
    flavor_blocks = [b for i,b in enumerate(blocks) if _ancestors(blocks,i) == ['android','productFlavors']]
    if flavor_blocks and not flavors and any(_contents(clean,b).strip() for b in flavor_blocks):
        problems.append('dynamic_or_unrecognized_product_flavors')
    for i,b in enumerate(blocks):
        if _ancestors(blocks,i) == ['android','buildTypes']:
            if re.search(r'\b(?:forEach|configureEach|register|named)\s*\(\s*[^\s"\']', _contents(clean,b)):
                problems.append('dynamic_build_types')
    if problems:
        variants, reason = [], 'incomplete_flavor_declarations_or_build_types'
    module_dir = posixpath.dirname(build_path)
    roots = _source_roots(clean, blocks, ai, module_dir)
    return {'build_file':build_path, 'module_root':module_dir, 'parser':'literal_gradle_dsl_scopes',
            'status':'declared_variant_candidates' if variants else (reason or 'unknown'),
            'flavor_dimensions':dimensions, 'build_types':build_types, 'flavors':flavors,
            'variants':variants, 'source_roots':roots, 'unresolved':problems + ([reason] if reason else [])}


def _files_for_source_set(path: str, mod: dict):
    """A source file must be under a known source set; unknown remains unknown."""
    root = mod.get('module_root', '')
    rel = path[len(root)+1:] if root and path.startswith(root+'/') else path if not root else ''
    if not rel:
        return None, 'outside_module'
    # Explicit Gradle source directories take precedence over default layout.
    roots = [x for x in mod.get('source_roots', []) if path.startswith(x['root']+'/')]
    if roots:
        longest = max(len(x['root']) for x in roots)
        owners = {x['source_set'] for x in roots if len(x['root']) == longest}
        if len(owners) == 1:
            return owners.pop(), 'explicit_literal_source_dir'
    if rel.startswith('src/'):
        bits = rel.split('/')
        if len(bits) >= 4 and bits[1]:
            return bits[1], 'standard_android_src_layout'
    return None, 'source_set_unknown'


def analyze_android_variants(root: Path, graph) -> dict[str,Any]:
    """Attach conservative variant hints to file nodes and graph edges.

    Run once AFTER symbol edges are added so overlays include those relations.
    Existing relation types and confidence are NEVER changed.
    """
    result: dict[str,Any] = {'schema':'atlas-android-variants/1', 'execution':'none',
        'claim':'declared_candidates_not_resolved_AGP_model', 'modules':{}, 'files':{},
        'edge_annotations':[], 'unresolved':[], 'note':
        'A disjoint pair is incompatible only within the enumerated literal candidate variants; '
        'this is not an assertion about evaluated Gradle builds or custom plugins.'}
    aliases = _plugin_aliases(root)
    for build_path in sorted(graph.nodes):
        if not build_path.endswith(('build.gradle', 'build.gradle.kts')):
            continue
        p = root / build_path
        try:
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding='utf8', errors='replace')
        except OSError:
            continue
        model = _module_model(build_path, text, aliases)
        if model:
            result['modules'][build_path] = model
            for issue in model.get('unresolved', []):
                result['unresolved'].append({'build_file':build_path, 'reason':issue})
    modules = result['modules']
    all_build_scripts = sorted((p for p in graph.nodes
                        if p.endswith(('build.gradle', 'build.gradle.kts'))),
                        key=lambda p: (-len(posixpath.dirname(p)),p))
    def owner(path):
        for build in all_build_scripts:
            base = posixpath.dirname(build)
            if not base or path.startswith(base + '/'):
                return modules.get(build)
        return None
    for path, node in graph.nodes.items():
        if not path.endswith(_SRC_EXT) or path.endswith(('build.gradle.kts', 'settings.gradle.kts')):
            continue
        mod = owner(path)
        if not mod:
            continue
        source_set, basis = _files_for_source_set(path,mod)
        variants = sorted(v['name'] for v in mod.get('variants', []) if source_set in v['source_sets']) if source_set else []
        status = ('source_set_candidates' if variants else 'unknown_or_unmatched_source_set')
        info = {'build_file':mod['build_file'], 'source_set':source_set,
                'basis':basis,'status':status, 'possible_variants':variants}
        result['files'][path] = info
        node.properties['android_source_set'] = source_set or ''
        node.properties['android_variant_status'] = status
        if variants:
            node.properties['android_candidate_variants'] = variants
        else:
            result['unresolved'].append({'file':path,'reason':'variant_membership_unresolved'})
    for e in graph.edges:
        left = result['files'].get(e.source)
        right = result['files'].get(e.target)
        if not left or not right or left['build_file'] != right['build_file']:
            continue
        a,b = left['possible_variants'],right['possible_variants']
        if not a or not b:
            continue
        if set(a).isdisjoint(b):
            e.properties['android_variant_compatibility'] = 'disjoint_declared_candidates'
            result['edge_annotations'].append({'from':e.source,'to':e.target,'type':e.type.value,
                'status':'disjoint_declared_candidates', 'note':'Inspect build model before treating this as a deployable relation'})
        else:
            e.properties['android_variant_compatibility'] = 'overlapping_declared_candidates'
    return result
