"""Local-only, non-evaluating deps.edn provenance for in-repository :local/root.

This is not tools.deps' effective basis: user/system deps, selected CLI args,
external Maven/Git coordinates and transitive conflict resolution are not
invented. No shell, project code, build script, or network call is performed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

MAX_LOCAL_LIBRARIES = 64
MAX_LOCAL_DEPTH = 10
MAX_LOCAL_BYTES = 256_000


class Unavailable(ValueError):
    pass


def _entries(form, *, keyword=False):
    if form is None:
        return {}
    if form.kind != 'map' or len(form.children)%2:
        raise Unavailable('expected_literal_dependency_map')
    result={}
    for i in range(0,len(form.children),2):
        k,v=form.children[i:i+2]
        if (k.kind!='atom' or not k.value or
                (k.value.startswith(':') != keyword) or k.value in result):
            raise Unavailable('invalid_or_duplicate_dependency_key')
        result[k.value]=v
    return result


def _root_inside_tree(root: Path, project: PurePosixPath, raw: str) -> PurePosixPath:
    if (not raw or '\x00' in raw or '\\' in raw or
            PurePosixPath(raw).is_absolute() or
            PureWindowsPath(raw).drive or PureWindowsPath(raw).is_absolute()):
        raise Unavailable('local_root_nonrelative_or_invalid')
    # Parent components are legitimate in local coordinates, but may never
    # escape the user's explicitly analyzed repository root.
    origin=root/project
    candidate=origin/PurePosixPath(raw)
    try:
        resolved=candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise Unavailable('local_root_does_not_exist') from None
    if not resolved.is_relative_to(root):
        raise Unavailable('local_root_outside_analyzed_tree')
    relative=resolved.relative_to(root)
    cursor=root
    for part in relative.parts:
        cursor=cursor/part
        if cursor.is_symlink():
            raise Unavailable('local_root_symlink')
    # Also reject symlink in a parent traversal that was normalized away.
    current=origin
    for part in PurePosixPath(raw).parts:
        current=current/part
        if current.is_symlink():
            raise Unavailable('local_root_symlink')
    if not resolved.is_dir():
        raise Unavailable('local_root_not_directory')
    return PurePosixPath(relative.as_posix())


def _decode_string(f):
    if f.kind!='string':
        raise Unavailable('local_root_expected_literal_string')
    from .clojure_shape import _decode_string as decode
    try:
        return decode(f.value)
    except (ValueError, TypeError):
        raise Unavailable('local_root_invalid_string') from None


def _edn_manifest(root: Path, rel: PurePosixPath):
    p=root/rel
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise Unavailable('local_deps_manifest_missing_or_unsafe')
    if p.stat().st_size>MAX_LOCAL_BYTES:
        raise Unavailable('local_deps_manifest_size_limit')
    raw=p.read_bytes()
    from .clojure_forms import read_forms
    forms=read_forms(raw.decode('utf-8-sig'))
    if len(forms)!=1:
        raise Unavailable('local_deps_manifest_expected_one_map')
    # Foreign tagged reader functions cannot be run under this static reader.
    from .clojure_source_context import _validate_edn
    _validate_edn(forms[0])
    return _entries(forms[0],keyword=True),hashlib.sha256(raw).hexdigest()


def local_roots(root: Path, project: PurePosixPath, root_manifest,
                selected_aliases=()):
    """Read declared local dependencies recursively, returning bounded roots.

    Does not claim that these roots are ordered or present in the actual JVM
    classpath. Each root retains a trace to its declaring manifest and lib.
    """
    roots=[];dependencies=[];issues=[];visited=set()
    def walk(project, manifest_data, manifest_rel, sha, depth):
        if depth>MAX_LOCAL_DEPTH:
            raise Unavailable('local_dependency_depth_limit')
        if manifest_rel in visited:
            raise Unavailable('local_dependency_cycle_or_shared_manifest')
        visited.add(manifest_rel)
        deps=_entries(manifest_data.get(':deps'),keyword=False)
        if depth==0 and selected_aliases:
            aliases=_entries(manifest_data.get(':aliases'),keyword=True)
            for alias in selected_aliases:
                conf=_entries(aliases.get(alias),keyword=True)
                if ':replace-deps' in conf:
                    deps=_entries(conf[':replace-deps'],keyword=False)
                if ':extra-deps' in conf:
                    deps.update(_entries(conf[':extra-deps'],keyword=False))
                if ':override-deps' in conf:
                    issues.append({'reason':'override_deps_not_effectively_resolved','alias':alias})
        for library, coord in deps.items():
            if coord.kind!='map':
                issues.append({'reason':'nonliteral_dependency_coordinate','library':library})
                continue
            fields=_entries(coord,keyword=True)
            if ':local/root' not in fields:
                issues.append({'reason':'external_dependency_unresolved','library':library})
                continue
            rel=_root_inside_tree(root,project,_decode_string(fields[':local/root']))
            dep_manifest=rel/'deps.edn'
            if len(dependencies)>=MAX_LOCAL_LIBRARIES:
                raise Unavailable('local_dependency_count_limit')
            dep_data,dep_sha=_edn_manifest(root,dep_manifest)
            record={'library':library,'root':rel.as_posix(),
                    'declaring_manifest':manifest_rel.as_posix(),
                    'declaring_manifest_sha256':sha,
                    'manifest':dep_manifest.as_posix(),
                    'manifest_sha256':dep_sha,
                    'dependency_coordinate_line':coord.line,
                    'dependency_coordinate_column':coord.col,
                    'status':'declared_local_dependency_not_effective_basis'}
            dependencies.append(record)
            ps=dep_data.get(':paths')
            if ps is None:
                issues.append({'reason':'local_dependency_paths_not_explicit',
                               'library':library,'manifest':dep_manifest.as_posix()})
            elif ps.kind not in ('vector','list'):
                raise Unavailable('invalid_local_dependency_paths')
            else:
                for item in ps.children:
                    segment=_decode_string(item)
                    path=_root_inside_tree(root,rel,segment)
                    roots.append({'path':path.as_posix(),'path_role':':local/root',
                                  'local_dependency':library,
                                  'local_dependency_evidence':record,
                                  'line':item.line,'column':item.col,
                                  'declared_value':segment,
                                  'declaring_cli_alias':None,
                                  'path_alias_chain':[]})
                    if len(roots)>MAX_LOCAL_LIBRARIES:
                        raise Unavailable('local_source_root_limit')
            walk(rel,dep_data,dep_manifest,dep_sha,depth+1)
    try:
        sha=hashlib.sha256((root/project/'deps.edn').read_bytes()).hexdigest()
        walk(project,root_manifest,project/'deps.edn',sha,0)
    except (Unavailable,UnicodeError,ValueError,OSError,RuntimeError,RecursionError) as exc:
        return {'status':'unavailable',
                'reason':str(exc)[:180] if isinstance(exc,Unavailable) else 'invalid_local_dependency_'+type(exc).__name__,
                'dependencies':dependencies,'roots':[],'limitations':issues,
                'effective_classpath_resolved':False}
    return {'status':'declared_local_dependency_evidence_only',
            'dependencies':dependencies,'roots':roots,'limitations':issues,
            'effective_classpath_resolved':False}
