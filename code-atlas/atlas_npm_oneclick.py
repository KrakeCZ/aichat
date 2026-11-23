"""Explicit, single-package download from pinned npm registry metadata.

Not used during analyze/startup. There is no npm install, arbitrary URL input,
project read, shell, plugin execution, or filesystem discovery. HTTPS metadata
from the OFFICIAL npm registry supplies the artifact's SHA-512. This is registry
provenance over TLS, not an independent cryptographic signature of the author.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from urllib.request import build_opener, HTTPRedirectHandler, Request

from atlas_npm_archive_import import PINNED_FILES, MAX_ARCHIVE, import_npm, registry_tarball, npm_name

MAX_META = 400_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError('Npm server redirected a request. Download canceled; inspect upstream URL.')


def metadata_url(package: str) -> str:
    if package not in PINNED_FILES:
        raise ValueError('Unknown/unsupported pinned graphics package')
    version = PINNED_FILES[package][0]
    return f'https://registry.npmjs.org/{quote(npm_name(package), safe="@")}/{version}'


def _read_fixed_url(url: str, *, limit: int, opener=None) -> bytes:
    # This function only receives URLs constructed from the fixed catalog.
    if not url.startswith('https://registry.npmjs.org/'):
        raise ValueError('Only official npm registry HTTPS URLs are allowed')
    transport = opener if opener is not None else build_opener(_NoRedirect())
    request = Request(url, headers={'User-Agent':'Universal-Code-Atlas/20 (manual graphics setup)',
                                    'Accept-Encoding':'identity', 'Accept':'application/json, application/octet-stream'})
    with transport.open(request, timeout=25) as response:
        if response.status != 200:
            raise ValueError(f'Unexpected registry HTTP status: {response.status}')
        # No silent redirects even when a custom test transport is injected.
        if response.geturl() != url:
            raise ValueError('Unexpected URL change (redirect rejected)')
        declared = response.headers.get('Content-Length')
        if declared and int(declared) > limit:
            raise ValueError('The upstream resource exceeds the fixed size limit')
        body = response.read(limit+1)
    if not body or len(body) > limit:
        raise ValueError('Invalid or oversized npm response')
    return body


def package_ready(manager, package: str) -> bool:
    if package not in PINNED_FILES:
        raise ValueError('Unknown pinned npm package')
    present, bad, records = manager.verify()
    expected = PINNED_FILES[package][1]
    return not any(n in bad for n in expected) and set(expected).issubset(present) and all(
        records['records'][n].get('category') == 'graphics'
        and records['records'][n].get('version') == PINNED_FILES[package][0]
        for n in expected)


def _package_storage(manager, name: str, version: str) -> Path:
    folder = manager.CACHE / 'packages'
    manager._cache_must_be_local()
    if folder.is_symlink() or not folder.resolve().is_relative_to(manager.CACHE.resolve()):
        raise ValueError('Unsafe packages cache directory')
    folder.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', name):
        raise ValueError('Unsafe npm package name')
    return folder / f'{name}-{version}.tgz'


def download_pinned_package(manager, package: str, *, allow_network: bool = False, opener=None) -> dict:
    """Download only after explicit consent. Cache success, skip all repeated requests.

    `opener` is for isolated tests only; production never accepts an opener URL
    from the client or project. Archive bytes stay inside Atlas's own cache.
    """
    if not allow_network:
        raise ValueError('Network consent is required; no network request was made')
    if package not in PINNED_FILES:
        raise ValueError('Package not in the pinned allowlist')
    if package_ready(manager, package):
        return {'package':package,'status':'cached','network_requests':0,
                'assets':sorted(PINNED_FILES[package][1])}

    version = PINNED_FILES[package][0]
    metadata = _read_fixed_url(metadata_url(package), limit=MAX_META, opener=opener)
    try:
        obj = json.loads(metadata.decode('utf8'))
        dist = obj['dist']
        integrity = dist['integrity']
        tarball = dist['tarball']
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError('Pinned npm registry metadata is missing required fields') from exc
    if obj.get('name') != npm_name(package) or obj.get('version') != version:
        raise ValueError('Unexpected package name/version in official registry metadata')
    if tarball != registry_tarball(package):
        raise ValueError('Npm tarball URL differs from pinned official registry URL')
    if not isinstance(integrity, str) or not re.fullmatch(r'sha512-[A-Za-z0-9+/]+={0,2}', integrity):
        raise ValueError('Official registry did not provide a SHA-512 integrity record')

    import base64
    try:
        expected = base64.b64decode(integrity[7:], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid registry SHA-512') from exc
    if len(expected) != 64:
        raise ValueError('Registry integrity hash length is invalid')
    archive = _read_fixed_url(tarball, limit=MAX_ARCHIVE, opener=opener)
    if hashlib.sha512(archive).digest() != expected:
        raise ValueError('SHA-512 mismatch between official registry metadata and archive')

    target = _package_storage(manager, npm_name(package).split('/')[-1], version)
    if target.is_symlink():
        raise ValueError('Refusing symlinked cached npm archive')
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tgz', delete=False) as f:
        tmp = Path(f.name)
    try:
        tmp.write_bytes(archive)
        # import_npm validates package contents/metadata/asset paths before
        # writing ANY graphics, so malicious tar paths cannot be extracted.
        result = import_npm(manager, tmp, integrity=integrity)
        if target.is_file() and target.read_bytes() != archive:
            raise ValueError('Different downloaded version already cached')
        if not target.exists():
            tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return {'package':package, 'version':version, 'status':'installed',
            'network_requests':2, 'assets':result['assets'], 'sha512_verified':True,
            'notice_available':result['notice_available'],
            'provenance':'Official npm registry metadata over HTTPS (not independent publisher signature)'}
