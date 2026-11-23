"""Import pinned, prebuilt graphics from explicitly selected npm .tgz files.

No npm, package scripts, tar extraction, network, project reads, or host search.
Only exact distribution filenames from the versioned allowlist are opened.
This importer is an offline conversion utility, not a dependency resolver.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

# name, pinned version, exact browser distribution path within the npm tarball.
# @highlightjs/cdn-assets is the browser distribution; highlight.js itself is
# mainly the source/server package and must not be substituted for CDN builds.
PINNED_FILES = {
    'cytoscape': ('3.34.3', {'cytoscape.min.js': 'dist/cytoscape.min.js'}),
    '@highlightjs/cdn-assets': ('11.11.1', {
        'highlight.min.js': 'highlight.min.js',
        'github-dark.min.css': 'styles/github-dark.min.css',
    }),
    'cytoscape-dagre': ('4.0.1', {'cytoscape-dagre.min.js': 'dist/cytoscape-dagre.min.js'}),
    'layout-base': ('2.0.1', {'layout-base.js': 'layout-base.js'}),
    'layout-base@1.0.2': ('1.0.2', {'layout-base-v1.js': 'layout-base.js'}),
    'avsdf-base': ('1.0.0', {'avsdf-base.js': 'avsdf-base.js'}),
    'cose-base': ('2.2.0', {'cose-base.js': 'cose-base.js'}),
    'cytoscape-fcose': ('2.2.0', {'cytoscape-fcose.js': 'cytoscape-fcose.js'}),
    'cytoscape-avsdf': ('1.0.0', {'cytoscape-avsdf.js': 'cytoscape-avsdf.js'}),
    '3d-force-graph': ('1.80.1', {'3d-force-graph.min.js': 'dist/3d-force-graph.min.js'}),
    'three-spritetext': ('1.10.0', {'three-spritetext.min.js': 'dist/three-spritetext.min.js'}),
}
MAX_ARCHIVE = 65_000_000
MAX_EXPANDED = 250_000_000
MAX_MEMBERS = 7000
MAX_SELECTED = 8_000_000
MAX_LICENSE = 300_000


def npm_name(package: str) -> str:
    """Actual upstream package name for a pinned version-qualified alias."""
    return 'layout-base' if package == 'layout-base@1.0.2' else package


def registry_tarball(package: str) -> str:
    """Public package tarball URL for manual user download; this never fetches it."""
    version, _ = PINNED_FILES[package]
    package = npm_name(package)
    if package.startswith('@'):
        scope, name = package[1:].split('/', 1)
        return f'https://registry.npmjs.org/@{scope}/{name}/-/{name}-{version}.tgz'
    return f'https://registry.npmjs.org/{package}/-/{package}-{version}.tgz'


def _npm_integrity_check(content: bytes, integrity: str | None, allow_unverified: bool) -> str:
    if not integrity:
        if not allow_unverified:
            raise ValueError('Trusted npm SHA-512 integrity required; alternatively explicitly use --accept-unverified-origin')
        return 'unverified_origin_explicitly_accepted'
    if not re.fullmatch(r'sha512-[A-Za-z0-9+/]+={0,2}', integrity):
        raise ValueError('Expected npm integrity syntax: sha512-BASE64')
    try:
        expected=base64.b64decode(integrity.removeprefix('sha512-'),validate=True)
    except ValueError as exc:
        raise ValueError('Malformed trusted SHA-512 integrity') from exc
    if len(expected)!=64 or hashlib.sha512(content).digest()!=expected:
        raise ValueError('Npm tarball SHA-512 mismatch — no resources imported')
    return 'verified_against_supplied_sha512'


def _read_selected(archive: bytes) -> tuple[str, str, dict[str, bytes], str | None]:
    """Collect only approved files in memory, never unpack archive paths."""
    try:
        tf=tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError('Not a readable npm .tgz archive') from exc
    with tf:
        selected: dict[str, bytes] = {}
        total=0
        seen=set()
        for index, member in enumerate(tf):
            if index>=MAX_MEMBERS:
                raise ValueError('Too many members in npm archive')
            # Repositories may contain extra files; do not traverse or extract any.
            if member.name in ('package', 'package/') and member.isdir():
                continue
            if not member.name.startswith('package/') or '\\' in member.name or '\x00' in member.name:
                raise ValueError('Unexpected npm archive entry location')
            components=member.name.split('/')
            if any(x in ('..','.','') for x in components[1:]):
                raise ValueError('Unsafe npm archive entry')
            if member.name in seen:
                raise ValueError('Duplicate entry in npm archive')
            seen.add(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Symlinks and special npm archive entries are not allowed')
            total+=member.size
            if total>MAX_EXPANDED:
                raise ValueError('Npm archive uncompressed size exceeds limit')
            name=member.name
            wanted = name == 'package/package.json' or name.startswith('package/LICENSE') or name.startswith('package/LICENCE') or name in {'package/COPYING'}
            if not wanted:
                # Exact allowlisted distribution paths, not generic JS/CSS files.
                wanted = any(name == 'package/'+relative for _, paths in PINNED_FILES.values() for relative in paths.values())
            if not wanted:
                continue
            limit = 60_000 if name == 'package/package.json' else MAX_LICENSE if name.startswith(('package/LICENSE','package/LICENCE')) or name == 'package/COPYING' else MAX_SELECTED
            if not member.isfile() or member.size<=0 or member.size>limit:
                raise ValueError('Invalid selected npm entry size/type')
            stream=tf.extractfile(member)
            if stream is None:
                raise ValueError('Npm entry is not an ordinary file')
            data=stream.read(limit+1)
            if len(data)!=member.size:
                raise ValueError('Npm entry read size mismatch')
            selected[name]=data
        if 'package/package.json' not in selected:
            raise ValueError('Npm package.json is missing')
        try:
            metadata=json.loads(selected.pop('package/package.json').decode('utf-8'))
        except (json.JSONDecodeError,UnicodeError) as exc:
            raise ValueError('Invalid npm package.json') from exc
        package=metadata.get('name')
        upstream_version = metadata.get('version')
        matches = [(key, pinned, paths) for key, (pinned, paths) in PINNED_FILES.items()
                   if npm_name(key) == package and pinned == upstream_version]
        if len(matches) != 1:
            raise ValueError('Unknown or unsupported graphics npm package/version')
        package_key, version, required = matches[0]
        from atlas_resources import GRAPHICS
        expected_licenses={GRAPHICS[output][1] for output in required}
        if metadata.get('license') not in expected_licenses:
            raise ValueError('Npm package license differs from pinned upstream license expectation')
        names={'package/'+relative: output for output,relative in required.items()}
        if any(name not in selected for name in names):
            raise ValueError('Npm archive lacks required pinned browser distribution: '+', '.join(sorted(n for n in names if n not in selected)))
        assets={name: selected[member] for member, name in names.items()}
        for name, body in assets.items():
            if len(body)<150 or b'<html' in body[:300].lower() or b'<!doctype' in body[:300].lower():
                raise ValueError('Browser distribution content too short or HTML: '+name)
        notices=[(name,blob) for name,blob in selected.items() if name in ('package/LICENSE','package/LICENSE.txt','package/LICENSE.md','package/LICENCE','package/LICENCE.md','package/COPYING')]
        license_text=notices[0][1].decode('utf-8',errors='replace') if notices else None
        return package_key,version,assets,license_text


def import_npm(manager, source: Path, *, integrity: str | None = None,
               accept_unverified_origin: bool = False) -> dict:
    """Import one .tgz selected by user. All validation precedes any cache write."""
    source=Path(source)
    if source.is_symlink() or not source.is_file() or source.suffix!='.tgz' or source.stat().st_size>MAX_ARCHIVE:
        raise ValueError('Expected one explicitly selected regular .tgz file (max 65MB)')
    body=source.read_bytes()
    if len(body)>MAX_ARCHIVE or len(body)<100:
        raise ValueError('Invalid npm archive size')
    provenance=_npm_integrity_check(body,integrity,accept_unverified_origin)
    package,version,assets,license_text=_read_selected(body)
    from atlas_resources import GRAPHICS
    if not set(assets).issubset(GRAPHICS):
        raise ValueError('Unexpected asset name in package')
    record=manager._manifest().get('records',{})
    for name,data in assets.items():
        existing=record.get(name)
        digest=hashlib.sha256(data).hexdigest()
        if existing and existing.get('sha256')!=digest:
            raise ValueError('Existing different version of '+name+'; no imports performed')
    with TemporaryDirectory(prefix='atlas-npm-import-') as temp:
        for name,content in sorted(assets.items()):
            path=Path(temp)/name
            path.write_bytes(content)
            manager.import_resource(name=name,source=path,category='graphics',
                expected_sha=hashlib.sha256(content).hexdigest(),
                origin=f'pinned-npm-archive:{package}@{version};{provenance}')
    # License notice is local provenance; if absent, disclose it rather than
    # silently claiming license compliance. No arbitrary tar entry paths used.
    notice_target=manager.CACHE/'licenses'/re.sub('[^A-Za-z0-9_.@-]', '_', package+'@'+version)
    if license_text:
        notice_target.parent.mkdir(parents=True,exist_ok=True)
        (notice_target.parent / (notice_target.name + '.LICENSE.txt')).write_text(license_text,encoding='utf-8')
    return {'package':package,'version':version,'assets':sorted(assets),
        'archive_sha256':hashlib.sha256(body).hexdigest(),
        'provenance':provenance,'notice_available':bool(license_text),
        'network':'disabled','project_access':'none'}
