"""Read-only exact-resource lookup in an explicitly supplied Clojure classpath.

The JSON snapshot is provided by the user AFTER computing a classpath (e.g.
using Clojure CLI in a separate trusted environment). Atlas must not invoke
Clojure CLI, fetch artifacts, evaluate build scripts, or guess missing entries.

This records what filesystem resources are present at a supplied classpath
ordering. It does NOT independently prove that this was the classpath used by
a compiler invocation, that cached jars are authentic, or that a symbol binds.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

MAX_ENTRIES=256
MAX_NAMESPACES=2000
MAX_MANIFEST_BYTES=300_000
MAX_JAR_BYTES=200_000_000
SUFFIXES=('.clj', '.cljc', '.cljs', '__init.class')


class SnapshotInvalid(ValueError):
    pass


def _unsafe_path(path):
    path=Path(path)
    if not path.is_absolute():return True
    # An explicit alias to a directory or jar may be a symlink; refuse
    # ambivalent identity rather than claiming a hash from another path.
    return path.is_symlink() or not path.exists() or path.resolve()!=path


def _hash_file(path, *, chunk=1024*1024):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            data=handle.read(chunk)
            if not data:break
            digest.update(data)
    return digest.hexdigest()


def load_snapshot(manifest_file, *, analyzed_root=None, selected_aliases=()):
    p=Path(manifest_file)
    if _unsafe_path(p) or not p.is_file() or p.stat().st_size>MAX_MANIFEST_BYTES:
        raise SnapshotInvalid('snapshot_missing_unsafe_or_too_large')
    raw=p.read_bytes()
    if p.suffix.lower() == '.basis':
        if analyzed_root is None:
            raise SnapshotInvalid('basis_requires_analyzed_root')
        from .clojure_effective_basis import read_basis_snapshot, BasisInvalid
        try:
            data=read_basis_snapshot(raw,path=p,analyzed_root=Path(analyzed_root),
                                     selected_aliases=selected_aliases)
        except (BasisInvalid, ValueError, OSError) as error:
            raise SnapshotInvalid(str(error)) from None
    else:
        try:data=json.loads(raw)
        except (ValueError,UnicodeError):raise SnapshotInvalid('snapshot_invalid_json') from None
    if not isinstance(data,dict) or data.get('schema')!='atlas-clojure-classpath-v1':
        raise SnapshotInvalid('snapshot_schema_unrecognized')
    entries=data.get('entries')
    if not isinstance(entries,list) or not 0<len(entries)<=MAX_ENTRIES:
        raise SnapshotInvalid('snapshot_entries_missing_or_too_many')
    if not isinstance(data.get('source'),str) or len(data['source'])>120:
        raise SnapshotInvalid('snapshot_source_missing')
    validated=[]
    for i,entry in enumerate(entries):
        if not isinstance(entry,dict) or entry.get('kind') not in ('jar','directory'):
            raise SnapshotInvalid('snapshot_entry_kind_invalid')
        raw_path=entry.get('path')
        if not isinstance(raw_path,str) or not raw_path:
            raise SnapshotInvalid('snapshot_entry_path_invalid')
        path=Path(raw_path)
        if _unsafe_path(path):
            raise SnapshotInvalid('snapshot_entry_missing_or_unsafe')
        if entry['kind']=='jar':
            if path.suffix.lower()!='.jar' or not path.is_file() or path.stat().st_size>MAX_JAR_BYTES:
                raise SnapshotInvalid('snapshot_jar_invalid')
            expected=entry.get('sha256')
            if not isinstance(expected,str) or len(expected)!=64 or expected!=_hash_file(path):
                raise SnapshotInvalid('snapshot_jar_hash_mismatch')
            try:
                with zipfile.ZipFile(path,'r') as z:
                    if len(z.namelist())>200_000:
                        raise SnapshotInvalid('snapshot_jar_entry_limit')
            except zipfile.BadZipFile:
                raise SnapshotInvalid('snapshot_jar_corrupt') from None
        else:
            if not path.is_dir():raise SnapshotInvalid('snapshot_directory_invalid')
        validated.append({'position':i, 'path':str(path), 'kind':entry['kind'],
                          'sha256':entry.get('sha256') if entry['kind']=='jar' else None,
                          'library':entry.get('library')})
    return {'status':'supplied_snapshot_validated_against_local_files',
            'source':data['source'], 'sha256':hashlib.sha256(raw).hexdigest(),
            'snapshot_path':str(p), 'entries':validated,
            'compiler_invocation_observed':False,
            'transitive_basis_resolved_by_atlas':False,
            'basis_evidence':data.get('basis_evidence')}


def _resource_has_symlink(classpath_root: Path, resource: str) -> bool:
    """Never silently skip a symlink in a requested resource's parent chain.

    The classpath root itself was validated by load_snapshot; the exact
    namespace resource may still be redirected by a nested symlink. A skipped
    earlier resource can make a later project file look like the loader's
    selected namespace, which is false linkage evidence.
    """
    current=classpath_root
    for part in resource.split('/'):
        if part in ('', '.', '..'):
            return True
        current=current/part
        if current.is_symlink():
            return True
    return False


class _JarProbeUnavailable(Exception):
    """No positive resource evidence may escape a failed archive validation."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _jar_identity(file_stat):
    return (file_stat.st_dev, file_stat.st_ino, file_stat.st_size,
            file_stat.st_mtime_ns, file_stat.st_ctime_ns)


def _hash_open_jar(handle):
    """Hash the SAME open file descriptor from which ZIP resources are read."""
    handle.seek(0)
    result = hashlib.sha256()
    total = 0
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > MAX_JAR_BYTES:
            raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
        result.update(block)
    return result.hexdigest()


@contextmanager
def _snapshot_jar(entry):
    """Verify a snapshot's JAR on the very descriptor used for ZIP inspection.

    A snapshot hash checked at load time does not prove the JAR is unchanged
    when namespace resolution runs later. Hashing a pathname and then opening
    it again also leaves a replace-between-opens race. Open a regular file
    once, compare its complete SHA-256 before inspection and again before
    publishing results. fstat plus path identity detects ordinary replacement
    or in-place modification during inspection. No claim of an adversarially
    atomic filesystem snapshot is made (e.g. hostile concurrent writes).
    """
    path = Path(entry['path'])
    expected = entry.get('sha256')
    if not isinstance(expected, str) or len(expected) != 64:
        raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
    try:
        if path.is_symlink() or path.resolve() != path:
            raise _JarProbeUnavailable('classpath_entry_unsafe_since_snapshot')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as handle:
            initial = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial.st_mode) or initial.st_size > MAX_JAR_BYTES:
                raise _JarProbeUnavailable('classpath_entry_unsafe_since_snapshot')
            opened_path = path.stat()
            if (opened_path.st_dev, opened_path.st_ino) != (initial.st_dev, initial.st_ino):
                raise _JarProbeUnavailable('classpath_entry_unsafe_since_snapshot')
            if _hash_open_jar(handle) != expected:
                raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
            if _jar_identity(os.fstat(handle.fileno())) != _jar_identity(initial):
                raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
            handle.seek(0)
            try:
                with zipfile.ZipFile(handle, 'r') as jar:
                    yield jar
            except (zipfile.BadZipFile, EOFError, RuntimeError, ValueError):
                raise _JarProbeUnavailable('classpath_jar_resource_unavailable') from None
            # Re-hash the opened archive after the ZIP directory is read;
            # never emit matches from a file modified during the scan.
            if _hash_open_jar(handle) != expected or (
                _jar_identity(os.fstat(handle.fileno())) != _jar_identity(initial)
            ):
                raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
            current = path.stat()
            if (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
                raise _JarProbeUnavailable('classpath_entry_hash_changed_since_snapshot')
    except (OSError, OverflowError):
        raise _JarProbeUnavailable('classpath_jar_resource_unavailable') from None


def probe_namespace(snapshot, namespace):
    """Probe exact namespace resources in classpath order, never call a loader."""
    from .clojure_source_context import namespace_resource
    resource=namespace_resource(namespace)
    if resource is None:
        return {'status':'unsupported_namespace_resource','candidates':[]}
    matches=[]
    blockers=[]
    for entry in snapshot['entries']:
        path=Path(entry['path'])
        if entry['kind']=='jar':
            try:
                with _snapshot_jar(entry) as z:
                    counts={}
                    wanted={resource+ext for ext in SUFFIXES}
                    for member in z.infolist():
                        if member.filename in wanted:
                            counts[member.filename]=counts.get(member.filename,0)+1
            except _JarProbeUnavailable as error:
                # Block ALL later providers: the earlier archive might contain
                # this namespace. Absence is not proven when its bytes changed.
                blockers.append({'position':entry['position'], 'resource':resource,
                                 'reason':error.code})
                continue
            for ext in SUFFIXES:
                target=resource+ext
                if counts.get(target,0)>1:
                    blockers.append({'position':entry['position'],'resource':target,
                                     'reason':'duplicate_jar_resource_name'})
                elif counts.get(target,0)==1:
                    matches.append({'position':entry['position'], 'entry_kind':'jar',
                                    'classpath_entry':entry['path'], 'resource':target,
                                    'sha256':entry['sha256']})
        else:
            for ext in SUFFIXES:
                rel=resource+ext
                if _resource_has_symlink(path,rel):
                    blockers.append({'position':entry['position'],'resource':rel,
                                     'reason':'symlink_in_namespace_resource_path'})
                    continue
                candidate=path/rel
                try:
                    if candidate.is_file() and candidate.resolve().is_relative_to(path.resolve()):
                        matches.append({'position':entry['position'], 'entry_kind':'directory',
                                        'classpath_entry':entry['path'],'resource':rel,
                                        'file_path':str(candidate.resolve())})
                except (OSError,RuntimeError):
                    blockers.append({'position':entry['position'],'resource':rel,
                                     'reason':'namespace_resource_probe_unavailable'})
    return {'status':'exact_resource_probe', 'candidates':matches,
            'blockers':blockers,
            'classpath_snapshot_sha256':snapshot['sha256']}


def inspect_snapshot(manifest_file, namespaces, analyzed_root, *, selected_aliases=()):
    snapshot=load_snapshot(manifest_file, analyzed_root=analyzed_root,
                           selected_aliases=selected_aliases)
    names=sorted(set(namespaces))
    if len(names)>MAX_NAMESPACES:
        raise SnapshotInvalid('namespace_probe_limit')
    from .clojure_source_context import namespace_resource
    indexed={name:namespace_resource(name) for name in names}
    probes={name:{'status':'exact_resource_probe','candidates':[], 'blockers':[],
                  'classpath_snapshot_sha256':snapshot['sha256']} for name in names}
    # Several distinct namespace spellings can map to one file path
    # (hyphen/underscore). Keep ALL probes, never let a dict overwrite one.
    wanted={}
    for name,resource in indexed.items():
        if resource:
            for suffix in SUFFIXES:
                wanted.setdefault(resource+suffix,[]).append(name)
    # Visit each jar exactly once. No full archive extraction, network access,
    # JVM class loading or recursive directory traversal takes place.
    for entry in snapshot['entries']:
        path=Path(entry['path'])
        if entry['kind']=='jar':
            try:
                with _snapshot_jar(entry) as jar:
                    counts={}
                    # Count actual ZIP members, not ambiguous NameToInfo keys.
                    for item in jar.infolist():
                        if item.filename in wanted:
                            counts[item.filename]=counts.get(item.filename,0)+1
            except _JarProbeUnavailable as error:
                # Unknown contents in an earlier classpath JAR prohibit a
                # positive relationship to any later namespace resource.
                for name,resource in indexed.items():
                    if resource:
                        probes[name]['blockers'].append({
                            'position':entry['position'], 'resource':resource,
                            'reason':error.code})
                continue
            for target,names in wanted.items():
                count=counts.get(target,0)
                if not count:continue
                for name in names:
                    if count>1:
                        probes[name]['blockers'].append({
                            'position':entry['position'], 'resource':target,
                            'reason':'duplicate_jar_resource_name'})
                    else:
                        probes[name]['candidates'].append({
                            'position':entry['position'],'entry_kind':'jar',
                            'classpath_entry':entry['path'], 'resource':target,
                            'sha256':entry['sha256']})
        else:
            for target,names in wanted.items():
                if _resource_has_symlink(path,target):
                    for name in names:
                        probes[name]['blockers'].append({
                            'position':entry['position'], 'resource':target,
                            'reason':'symlink_in_namespace_resource_path'})
                    continue
                candidate=path/target
                try:
                    if candidate.is_file() and candidate.resolve().is_relative_to(path.resolve()):
                        for name in names:
                            probes[name]['candidates'].append({
                                'position':entry['position'],'entry_kind':'directory',
                                'classpath_entry':entry['path'],'resource':target,
                                'file_path':str(candidate.resolve())})
                except (OSError,RuntimeError):
                    for name in names:
                        probes[name]['blockers'].append({
                            'position':entry['position'], 'resource':target,
                            'reason':'namespace_resource_probe_unavailable'})
    return {'status':snapshot['status'], 'source':snapshot['source'],
            'sha256':snapshot['sha256'], 'entries':snapshot['entries'],
            'resource_probes':probes,
            'compiler_invocation_observed':False,
            'transitive_basis_resolved_by_atlas':False,
            'analyzed_root':str(Path(analyzed_root).resolve()),
            'basis_evidence':snapshot.get('basis_evidence')}


def public_snapshot_evidence(snapshot, analyzed_root):
    """Remove absolute local machine paths from exportable graph JSON.

    Full paths stay only in the in-memory resolver, never in generated web
    files which a user may later publish. Retain resource name, order, hashes,
    and project-relative source paths for auditability.
    """
    root=Path(analyzed_root).resolve()
    output={k:v for k,v in snapshot.items() if k not in
            ('entries','resource_probes','analyzed_root','snapshot_path')}
    output['entries']=[{'position':e['position'], 'kind':e['kind'],
                        'basename':Path(e['path']).name,'sha256':e['sha256'],
                        'library':e.get('library')}
                       for e in snapshot['entries']]
    probes={}
    for namespace,record in snapshot['resource_probes'].items():
        candidates=[]
        for c in record.get('candidates',[]):
            view={k:v for k,v in c.items() if k not in ('classpath_entry','file_path')}
            view['classpath_entry_basename']=Path(c['classpath_entry']).name
            if 'file_path' in c:
                source=Path(c['file_path'])
                view['project_relative_file']=(source.relative_to(root).as_posix()
                                               if source.is_relative_to(root) else None)
            candidates.append(view)
        probes[namespace]={**record,'candidates':candidates}
    output['resource_probes']=probes
    return output
