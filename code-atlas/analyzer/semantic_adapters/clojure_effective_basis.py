"""Read a Clojure tools.deps *already computed* .basis file, without execution.

The Clojure CLI's :classpath-roots are an ordered result of dependency
resolution (including transitive Maven/Git libraries).  We do not derive that
order from deps.edn or use this file as proof of a fresh compiler invocation.
All resources are opened locally, with bounded parsing and no reader-eval.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .clojure_source_context import normalize_cli_aliases


class BasisInvalid(ValueError):
    pass


def _map(form, *, keyword_keys=True):
    if form.kind != 'map' or form.metadata or len(form.children) % 2:
        raise BasisInvalid('basis_expected_map')
    result = {}
    for key, val in zip(form.children[::2], form.children[1::2]):
        if key.kind not in ('atom', 'string') or key.metadata:
            raise BasisInvalid('basis_invalid_map_key')
        if keyword_keys and (key.kind != 'atom' or not key.value.startswith(':') or key.value.startswith('::')):
            raise BasisInvalid('basis_expected_keyword_key')
        if key.value in result:
            raise BasisInvalid('basis_duplicate_map_key')
        result[key.value] = val
    return result


def _text(form):
    if form.kind != 'string' or form.metadata or not isinstance(form.value, str):
        raise BasisInvalid('basis_expected_string')
    from .clojure_shape import _decode_string
    try:
        return _decode_string(form.value)
    except ValueError as exc:
        raise BasisInvalid('basis_invalid_string') from exc


def read_basis_snapshot(raw: bytes, *, path: Path, analyzed_root: Path, selected_aliases=()):
    """Turn a real CLI basis into the *existing* classpath snapshot contract.

    No project classloading or Clojure CLI execution is performed. Directory
    snapshots intentionally have no source-content freshness guarantee.
    """
    from .clojure_forms import read_forms, ReaderError
    from .clojure_classpath_snapshot import MAX_ENTRIES, MAX_JAR_BYTES, _hash_file
    if len(raw) > 300_000:
        raise BasisInvalid('basis_size_limit')
    try:
        forms = read_forms(raw.decode('utf-8-sig'))
    except (ReaderError, UnicodeError, ValueError) as exc:
        raise BasisInvalid('basis_invalid_edn') from exc
    if len(forms) != 1:
        raise BasisInvalid('basis_expected_one_form')
    data = _map(forms[0])
    config = _map(data[':basis-config']) if ':basis-config' in data else None
    if config is None or ':dir' not in config:
        raise BasisInvalid('basis_missing_origin_directory')
    dir_text = _text(config[':dir'])
    origin = Path(dir_text)
    root = Path(analyzed_root).resolve()
    if not origin.is_absolute():
        # CLI permits a relative :basis-config :dir. In a project .cpcache
        # it is relative to the project working directory, NOT the cache
        # folder. Refuse any other guessed location.
        cache_parent = path.parent
        anchor = cache_parent.parent if cache_parent.name == '.cpcache' else root
        origin = anchor/origin
    origin = origin.resolve()
    if not origin.is_relative_to(root) or not (origin/'deps.edn').is_file():
        raise BasisInvalid('basis_origin_not_analyzed_project')
    configured = config.get(':aliases')
    if configured is None:
        aliases = ()
    elif configured.kind != 'vector' or configured.metadata or any(
            item.kind != 'atom' or not item.value.startswith(':') for item in configured.children):
        raise BasisInvalid('basis_aliases_invalid')
    else:
        aliases = normalize_cli_aliases([item.value for item in configured.children])
    requested = normalize_cli_aliases(selected_aliases)
    if tuple(requested) != tuple(aliases):
        raise BasisInvalid('basis_aliases_do_not_match_selected_scenario')
    roots = data.get(':classpath-roots')
    if roots is None or roots.kind != 'vector' or roots.metadata or not (0 < len(roots.children) <= MAX_ENTRIES):
        raise BasisInvalid('basis_classpath_roots_missing_or_too_many')
    listed = [_text(item) for item in roots.children]
    if len(set(listed)) != len(listed):
        raise BasisInvalid('basis_duplicate_classpath_roots')
    cp = _map(data[':classpath'], keyword_keys=False) if ':classpath' in data else None
    if cp is not None and any(item not in cp for item in listed):
        raise BasisInvalid('basis_root_not_in_classpath_map')
    libs = _map(data[':libs'], keyword_keys=False) if ':libs' in data else {}
    if len(libs) > 10000:
        raise BasisInvalid('basis_dependency_limit')
    entries = []
    for item in listed:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = origin/candidate
        # Do not normalize a symlink into a different identity: every entry is
        # a literal file system claim from an existing basis.
        if candidate.is_symlink() or candidate.resolve() != candidate or not candidate.exists():
            raise BasisInvalid('basis_classpath_entry_missing_or_symlink')
        if candidate.is_dir():
            entry = {'kind':'directory', 'path':str(candidate)}
        elif candidate.is_file() and candidate.suffix.lower() == '.jar':
            if candidate.stat().st_size > MAX_JAR_BYTES:
                raise BasisInvalid('basis_jar_size_limit')
            entry = {'kind':'jar', 'path':str(candidate), 'sha256':_hash_file(candidate)}
        else:
            raise BasisInvalid('basis_entry_not_jar_or_directory')
        if cp is not None:
            info = _map(cp[item])
            lib = info.get(':lib-name')
            if lib and lib.kind == 'atom':
                entry['library'] = lib.value
        entries.append(entry)
    return {'schema':'atlas-clojure-classpath-v1',
            'source':'clojure.tools.deps_precomputed_basis_read_only',
            'entries':entries,
            'basis_evidence':{
                'source_sha256':hashlib.sha256(raw).hexdigest(),
                'classpath_root_count':len(entries),
                'dependency_count':len(libs),
                'basis_aliases':list(aliases),
                'origin_project_relative':origin.relative_to(root).as_posix(),
                'project_manifest_sha256':hashlib.sha256((origin/'deps.edn').read_bytes()).hexdigest(),
                'freshness_against_project_manifest_unproven':True,
                'compiler_invocation_observed':False,
                'maven_git_resolution_executed_by_atlas':False}}
