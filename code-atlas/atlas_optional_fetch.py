"""Explicit user-initiated public asset download ONLY; never called by analyze.

No source/project path is sent. Only exact versioned public URL from the Atlas
catalogue is contacted after --allow-network. No arbitrary URL or redirects.
"""
from __future__ import annotations
import hashlib
import io
from urllib.request import build_opener, HTTPRedirectHandler, Request
from urllib.error import HTTPError, URLError

from prepare_visual_assets import ASSETS

MAX_BYTES=8_000_000

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError('Redirect refused; inspect upstream source manually')


def fetch_graphics_asset(manager, *, name: str, allow_network: bool = False,
                         expected_sha: str | None = None, opener=None) -> dict:
    if not allow_network:
        raise ValueError('Network is disabled. Explicit --allow-network is required.')
    url=ASSETS.get(name)
    if not url or not url.startswith('https://'):
        raise ValueError('Only an allowlisted versioned official graphics URL is permitted')
    if expected_sha and (len(expected_sha)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected_sha)):
        raise ValueError('Invalid supplied expected SHA-256')
    records=manager._manifest().get('records',{})
    if name in records:
        ok,bad,_=manager.verify()
        if name in ok:
            raise ValueError('Already cached and verified; no repeated network access. Use status.')
        if name in bad:
            raise ValueError('Invalid cached file. Remove/repair it explicitly before a new download.')
    client=opener or build_opener(_NoRedirect())
    req=Request(url,headers={'User-Agent':'Universal-Code-Atlas-ManualResourceManager/1','Accept':'text/javascript,text/css,application/javascript,*/*'})
    # Network I/O only here, on explicit CLI action. No project paths or files.
    with client.open(req,timeout=20) as response:
        body=response.read(MAX_BYTES+1)
    if len(body)<150 or len(body)>MAX_BYTES or b'<html' in body[:300].lower() or b'<!doctype' in body[:300].lower():
        raise ValueError('Invalid / oversized public asset response; cache unchanged')
    sha=hashlib.sha256(body).hexdigest()
    if expected_sha and sha!=expected_sha.lower():
        raise ValueError('Downloaded SHA-256 differs from user-provided trusted digest')
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    manager.CACHE.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(prefix='atlas-web-asset-',dir=manager.CACHE,delete=False) as temp:
        tmp=Path(temp.name);temp.write(body)
    try:
        rec=manager.import_resource(name=name,source=tmp,category='graphics',expected_sha=expected_sha,
                                    origin=url)
    finally:
        tmp.unlink(missing_ok=True)
    return {'name':name,'sha256':rec['sha256'],'bytes':rec['bytes'],
            'source_url':url,'upstream_digest_verified':bool(expected_sha),
            'notice':'Hash ensures stored byte integrity; without external trusted checksum, upstream authenticity is not established.'}
