#!/usr/bin/env python3
"""Universal Code Atlas local resource inventory; offline unless user explicitly acts.

All reads are restricted to the Atlas installation and files supplied by an
explicit --file CLI argument. No home-directory, registry or PATH scanning.
The original 2D/3D renderer remains unchanged: activating the complete local
set populates its pre-existing `vendor` compatibility directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

BASE = Path(__file__).resolve().parent
CACHE = BASE / 'online-resources'
MANIFEST = CACHE / 'manifest.json'
SCHEMA = 'atlas-offline-resources/1'
GRAPHICS = {
    'cytoscape.min.js': ('3.34.3', 'MIT'),
    'highlight.min.js': ('11.11.1', 'BSD-3-Clause'),
    'github-dark.min.css': ('11.11.1', 'BSD-3-Clause'),
    'cytoscape-dagre.min.js': ('4.0.1', 'MIT'),
    'layout-base.js': ('2.0.1', 'MIT'),
    'cose-base.js': ('2.2.0', 'MIT'),
    'cytoscape-fcose.js': ('2.2.0', 'MIT'),
    'layout-base-v1.js': ('1.0.2', 'MIT'),
    'avsdf-base.js': ('1.0.0', 'MIT'),
    'cytoscape-avsdf.js': ('1.0.0', 'MIT'),
    '3d-force-graph.min.js': ('1.80.1', 'MIT'),
    'three-spritetext.min.js': ('1.10.0', 'MIT'),
}


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _cache_must_be_local() -> None:
    # The resource manager is intentionally limited to Atlas's own directory.
    if CACHE.is_symlink() or not CACHE.resolve().is_relative_to(BASE.resolve()):
        raise ValueError('Atlas resource cache must remain inside its installation')
    for subdir in (CACHE/'blobs', CACHE/'licenses'):
        if subdir.is_symlink():
            raise ValueError('Symlinked resource storage is not supported')


def _manifest() -> dict:
    _cache_must_be_local()
    if not MANIFEST.is_file():
        return {'schema': SCHEMA, 'records': {}}
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or not isinstance(data.get('records'), dict):
        raise ValueError('Unknown or malformed offline resources manifest')
    return data


def _atomic_manifest(data: dict) -> None:
    _cache_must_be_local()
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / '.manifest.tmp'
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(MANIFEST)


def _resource_path(sha: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{64}', sha):
        raise ValueError('Invalid SHA-256 in resource manifest')
    _cache_must_be_local()
    target = CACHE / 'blobs' / sha[:2] / sha
    if (CACHE/'blobs'/sha[:2]).is_symlink():
        raise ValueError('Symlinked blob directory is not allowed')
    return target


def verify() -> tuple[list[str], list[str], dict]:
    data = _manifest()
    ok, bad = [], []
    for name, info in sorted(data['records'].items()):
        if not isinstance(info, dict) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', name):
            bad.append(name)
            continue
        try:
            path = _resource_path(info['sha256'])
            # A blob is never an executable loaded from an arbitrary path.
            if path.is_symlink() or not path.is_file() or _hash(path) != info['sha256']:
                bad.append(name)
            else:
                ok.append(name)
        except (OSError, KeyError, TypeError, ValueError):
            bad.append(name)
    return ok, bad, data


def import_resource(*, name: str, source: Path, category: str,
                    expected_sha: str | None = None, origin: str = '') -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', name):
        raise ValueError('Resource name must be a simple filename, not a path')
    if category not in ('graphics', 'toolchain', 'grammar'):
        raise ValueError('Unknown category')
    if category == 'graphics' and name not in GRAPHICS:
        raise ValueError('Not a recognized Atlas graphics asset')
    if category != 'graphics' and not expected_sha:
        raise ValueError('Toolchains and grammars require an expected SHA-256')
    # This file was selected by the user explicitly via --file. No directory walk.
    source = Path(source)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 250_000_000:
        raise ValueError('Explicit source must be one regular file, max 250MB')
    sha = _hash(source)
    if expected_sha and expected_sha.lower() != sha:
        raise ValueError('SHA-256 mismatch; resource was not installed')
    data = _manifest()
    if name in data['records'] and data['records'][name]['sha256'] != sha:
        raise ValueError('Different version exists. Use a distinct versioned name.')
    target = _resource_path(sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as writer:
            tmp = Path(writer.name)
        try:
            shutil.copyfile(source, tmp)
            if _hash(tmp) != sha:
                raise ValueError('Source changed during import')
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    size = target.stat().st_size
    data['records'][name] = {'sha256': sha, 'bytes': size,
                             'category': category, 'origin': origin,
                             'expected_sha_provided': bool(expected_sha),
                             'version': GRAPHICS[name][0] if category == 'graphics' else '',
                             'license_hint': GRAPHICS[name][1] if category == 'graphics' else 'see upstream license'}
    _atomic_manifest(data)
    return data['records'][name]


def activate_graphics() -> list[str]:
    """Activate ONLY the complete expected set, preserving original UI logic."""
    ok, bad, data = verify()
    present = {name for name in ok if data['records'][name].get('category') == 'graphics'}
    missing = sorted(set(GRAPHICS) - present)
    if bad or missing:
        raise ValueError('Not activated. Missing: ' + ', '.join(missing) + '; invalid: ' + ', '.join(bad))
    # An existing manifest can contain additional non-graphics tools. Only copy
    # the known versioned graphics assets; avoid teaching Atlas to load unknown tools.
    from prepare_visual_assets import ASSETS
    vendor = BASE / 'vendor'
    if vendor.is_symlink():
        raise ValueError('Symlinked vendor directory is not supported')
    vendor.mkdir(exist_ok=True)
    converted = {}
    for name in GRAPHICS:
        sha = data['records'][name]['sha256']
        target = vendor / name
        src = _resource_path(sha)
        if target.is_symlink():
            raise ValueError('Refusing symlinked vendor asset')
        if not target.is_file() or _hash(target) != sha:
            shutil.copyfile(src, target)
        converted[name] = {'url': ASSETS[name], 'sha256': sha, 'bytes': target.stat().st_size}
    (vendor/'manifest.json').write_text(json.dumps(converted, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return list(GRAPHICS)


def import_legacy_vendor() -> list[str]:
    """Import ONLY Atlas's own legacy vendor folder; never searches user disks."""
    from prepare_visual_assets import ASSETS
    vendor = BASE / 'vendor'
    mf = vendor / 'manifest.json'
    if not mf.is_file():
        raise ValueError('No local legacy vendor manifest found')
    old = json.loads(mf.read_text(encoding='utf-8'))
    verified = {}
    for name, url in ASSETS.items():
        # A Stage-21 vendor installation correctly contains ten assets.
        # Import those ten without claiming the two AVSDF transitives exist.
        if name not in old:
            continue
        rec = old[name]
        f = vendor / name
        if rec.get('url') != url or f.is_symlink() or not f.is_file() or not re.fullmatch(r'[a-f0-9]{64}',rec.get('sha256','')):
            raise ValueError('Missing or malformed legacy graphic: ' + name)
        if _hash(f) != rec['sha256']:
            raise ValueError('Legacy vendor graphics SHA-256 mismatch: ' + name)
        verified[name] = (f, rec['sha256'], url)
    for name, (f, sha, url) in verified.items():
        import_resource(name=name, source=f, category='graphics', expected_sha=sha, origin=url)
    return list(verified)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Atlas local resources: no automatic downloads; explicit one-click setup is optional.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('manager', help='Open one-click LOCAL graphics manager; no network without a click')
    status = sub.add_parser('status', help='Show installed and missing assets; no network')
    status.add_argument('--json', action='store_true')
    add = sub.add_parser('import', help='Copy one explicitly chosen local file to content-addressed cache')
    add.add_argument('--name', required=True)
    add.add_argument('--file', required=True, type=Path)
    add.add_argument('--category', choices=('graphics','toolchain','grammar'), default='graphics')
    add.add_argument('--sha256', default=None, help='Expected upstream SHA-256 (required for toolchains/grammars)')
    add.add_argument('--origin', default='', help='Provenance label or original URL, never fetched')
    sub.add_parser('activate-graphics', help='Copy complete local graphics cache to existing vendor folder')
    sub.add_parser('import-legacy-vendor', help='Migrate own existing vendor/manifest.json to online-resources (no network)')
    cat = sub.add_parser('catalog', help='Official links, versions, required files and local cache; no network')
    cat.add_argument('--json', action='store_true')
    guide = sub.add_parser('instructions', help='Manual download, copy and verification instructions; no network')
    guide.add_argument('--all', action='store_true', help='Include already installed resources')
    exp = sub.add_parser('pack-create', help='Create portable USB ZIP of explicitly installed resources; offline')
    exp.add_argument('--output', required=True, type=Path)
    exp.add_argument('--name', action='append', default=[], help='Resource name; may be specified more than once. Default: all cached.')
    imp = sub.add_parser('pack-install', help='Import a portable offline ZIP, verify hashes and never execute any contents')
    imp.add_argument('--file', required=True, type=Path)
    npm = sub.add_parser('import-npm', help='Import prebuilt browser files from one pinned npm tarball (offline; no npm install)')
    npm.add_argument('--file',required=True,type=Path, help='Exact local .tgz selected by the user')
    npm.add_argument('--integrity', help='Trusted sha512-BASE64 digest of npm tarball (e.g. from trusted package lock)')
    npm.add_argument('--accept-unverified-origin', action='store_true', help='Explicitly accept archive without independently trusted SHA-512; origin is NOT verified')
    sub.add_parser('npm-packages', help='Exact pinned tarball URLs and expected file names; no network')
    fetch = sub.add_parser('fetch-graphics', help='Optional one-file network transfer, ONLY when explicitly approved')
    fetch.add_argument('--name', required=True, choices=sorted(GRAPHICS))
    fetch.add_argument('--allow-network', action='store_true',help='Explicit consent for one fixed upstream public URL; no source code is sent')
    fetch.add_argument('--sha256',default=None,help='Trusted SHA-256, recommended; if absent origin is not independently verified')


    args = ap.parse_args(argv)
    try:
        if args.cmd == 'manager':
            from atlas_resource_http import serve_standalone
            serve_standalone(open_browser=True)
            return 0
        if args.cmd == 'status':
            ok, bad, data = verify()
            present = {name for name in ok if data['records'][name].get('category') == 'graphics'}
            missing = sorted(set(GRAPHICS) - present)
            report = {'network': 'disabled', 'cache_path': str(CACHE),
                      'installed': ok, 'invalid': bad, 'missing_graphics': missing,
                      'graphics_ready': not missing and not bad,
                      'records': data['records']}
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(f'Atlas offline cache: {len(ok)} installed / {len(GRAPHICS)} graphics required')
                print('No network operations. Cache:', CACHE)
                print('Missing graphics:', ', '.join(missing) if missing else 'none')
                if bad: print('Invalid:', ', '.join(bad))
            return 1 if bad else 0
        if args.cmd in ('catalog', 'instructions'):
            from atlas_resource_catalog import inventory, instructions
            ok,bad,data=verify()
            report=inventory(ok,bad,data['records'])
            if args.cmd == 'instructions':
                print(instructions(report,missing_only=not args.all),end='')
            elif args.json:
                print(json.dumps(report,ensure_ascii=False,indent=2))
            else:
                print(f'Atlas graphics: {report["summary"]["present"]}/{report["summary"]["required"]}; no network')
                for item in report['graphics']:
                    state='OK' if item['installed'] else 'INVALID' if item['invalid'] else 'MISSING'
                    print(state,item['name'],item['version_hint'],item['official_package'])
                print('Details: python atlas_resources.py instructions')
            return 1 if bad else 0
        if args.cmd == 'fetch-graphics':
            from atlas_optional_fetch import fetch_graphics_asset
            result=fetch_graphics_asset(sys.modules[__name__],name=args.name,
                allow_network=args.allow_network,expected_sha=args.sha256)
            print('One explicit download completed:',json.dumps(result,ensure_ascii=False))
            return 0
        if args.cmd == 'npm-packages':
            from atlas_npm_archive_import import PINNED_FILES, registry_tarball
            for package, (version, files) in PINNED_FILES.items():
                print(f'{package}@{version}')
                print('  Official npm tarball: '+registry_tarball(package))
                print('  Provides: '+', '.join(sorted(files)))
                print('  Import: python atlas_resources.py import-npm --file '+
                      f'online-resources/inbox/{package.rsplit("/",1)[-1]}-{version}.tgz --integrity sha512-<TRUSTED_BASE64>')
            return 0
        if args.cmd == 'import-npm':
            from atlas_npm_archive_import import import_npm
            result=import_npm(sys.modules[__name__],args.file,integrity=args.integrity,
                              accept_unverified_origin=args.accept_unverified_origin)
            print('Offline npm package imported:',json.dumps(result,ensure_ascii=False,indent=2))
            return 0
        if args.cmd == 'pack-create':
            from atlas_resource_pack import create_pack
            result=create_pack(sys.modules[__name__],args.output,args.name)
            print('Pack created (offline):',json.dumps(result,ensure_ascii=False))
            return 0
        if args.cmd == 'pack-install':
            from atlas_resource_pack import install_pack
            result=install_pack(sys.modules[__name__],args.file)
            print('Pack imported (offline):',json.dumps(result,ensure_ascii=False))
            return 0
        if args.cmd == 'import':
            result = import_resource(name=args.name, source=args.file, category=args.category,
                                     expected_sha=args.sha256, origin=args.origin)
            print(f'Cached {args.name}: {result["bytes"]} bytes, SHA-256 {result["sha256"]}')
        elif args.cmd == 'activate-graphics':
            print('Activated graphics (without internet):', ', '.join(activate_graphics()))
        elif args.cmd == 'import-legacy-vendor':
            print('Imported existing local graphics (no network):', ', '.join(import_legacy_vendor()))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print('Offline resources:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
