"""Loopback-only, explicit-click resource management for the existing Atlas server.

The user must click a button in locally served HTML before any public npm
transfer. A separate session token, same-origin check and fixed npm allowlist
block cross-site requests and arbitrary URLs. No project data is included.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import secrets
import shutil
import tempfile
import threading
from urllib.parse import urlsplit

from atlas_npm_archive_import import PINNED_FILES, MAX_ARCHIVE, import_npm
from atlas_npm_oneclick import download_pinned_package, package_ready
from atlas_resources import GRAPHICS


def configure_server(server):
    """Called once per Atlas localhost HTTP server, never by analyze()."""
    server.atlas_resources_token = secrets.token_urlsafe(36)
    server.atlas_resources_lock = threading.Lock()


def _response(handler, code: int, data: dict) -> None:
    raw = json.dumps(data, ensure_ascii=False).encode('utf8')
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(raw)))
    handler.send_header('Cache-Control', 'no-store')
    handler.send_header('X-Content-Type-Options', 'nosniff')
    handler.send_header('Content-Security-Policy', "default-src 'none'")
    handler.end_headers()
    handler.wfile.write(raw)


def _trusted(handler, require_origin=False) -> bool:
    port=handler.server.server_address[1]
    base=f'http://127.0.0.1:{port}'
    if handler.client_address[0] not in ('127.0.0.1','::1'):
        return False
    if handler.headers.get('Host') != f'127.0.0.1:{port}':
        return False
    if require_origin:
        if handler.headers.get('Origin') != base:
            return False
        if handler.headers.get('Sec-Fetch-Site') not in (None,'same-origin'):
            return False
        if not secrets.compare_digest(handler.headers.get('X-Atlas-Token',''),handler.server.atlas_resources_token):
            return False
    return True


def _inventory():
    from atlas_resources import verify
    from atlas_resource_catalog import inventory
    from atlas_npm_archive_import import PINNED_FILES
    ok,bad,data=verify()
    result=inventory(ok,bad,data['records'])
    result['packages']=[{'name':package,'version':version,'installed':package_ready(__import__('atlas_resources'),package),
                         'assets':sorted(files)} for package,(version,files) in PINNED_FILES.items()]
    return result


def _sync_site(site:Path) -> None:
    """Make installed graphics usable in the *already generated* offline site."""
    from atlas_resources import verify
    from atlas_resource_catalog import html_report
    from atlas_npm_archive_import import PINNED_FILES
    from prepare_visual_assets import ASSETS
    ok,bad,data=verify()
    if bad or not set(GRAPHICS).issubset(ok):
        return
    # A site built with Stage 21 still has the old single-script AVSDF loader.
    # Patch that one known JS file when the user explicitly installs its new
    # dependencies, without rescanning/rebuilding the project.
    from atlas_avsdf_hotfix import repair as repair_avsdf
    renderer = site / 'atlas-view-modes.js'
    if renderer.is_file():
        try:
            repair_avsdf(site)
        except (OSError, ValueError) as exc:
            # Do not discard the installed graphics if a user customized JS.
            print('AVSDF site patch skipped:', exc)
    if (site/'online-resources').is_symlink():
        raise ValueError('Refusing symlinked site resources directory')
    folder=site/'online-resources'/'graphics'
    if folder.is_symlink():
        raise ValueError('Refusing symlinked site graphics directory')
    folder.mkdir(parents=True,exist_ok=True)
    for name in GRAPHICS:
        sha=data['records'][name]['sha256']
        from atlas_resources import _resource_path
        source=_resource_path(sha)
        target=folder/name
        if target.is_symlink():
            raise ValueError('Refusing symlinked site asset')
        shutil.copyfile(source,target)
    manifest={'schema':'atlas-offline-site-assets/1','network':'no_automatic_graphics_requests',
        'files':{name:{'sha256':data['records'][name]['sha256'], 'upstream_url':ASSETS[name],
                       'version_hint':GRAPHICS[name][0], 'license_hint':GRAPHICS[name][1]}
                 for name in GRAPHICS}}
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    from atlas_resources import CACHE
    root=CACHE/'licenses'
    dest=site/'online-resources'/'licenses'
    notices=[]
    for package,(version,assets) in PINNED_FILES.items():
        safe=__import__('re').sub(r'[^A-Za-z0-9_.@-]','_',f'{package}@{version}')+'.LICENSE.txt'
        file=root/safe
        if file.is_file() and not file.is_symlink() and file.stat().st_size < 300_000:
            if dest.is_symlink():
                raise ValueError('Refusing symlinked site license directory')
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(file,dest/safe)
            notices.append({'package':package,'version':version,'file':f'online-resources/licenses/{safe}',
                            'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    if notices:
        (dest/'manifest.json').write_text(json.dumps({'schema':'atlas-upstream-license-notices/1','notices':notices},indent=2)+'\n',encoding='utf8')
    result=_inventory()
    result['summary']['renderer_assets_ready']=True
    result['summary']['missing_renderer_assets']=[]
    result['summary']['output_policy']='offline_default'
    (site/'resource-status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (site/'resource-manager.html').write_text(html_report(result),encoding='utf8')
    index=site/'index.html'
    if index.is_file():
        data=index.read_text(encoding='utf8')
        updated=data.replace('Offline grafika zatím chybí — zdrojová analýza funguje; pro 2D/3D přidej lokální knihovny',
                             'Offline grafika připravena')
        if data!=updated:
            index.write_text(updated,encoding='utf8')


def get(handler) -> bool:
    path=urlsplit(handler.path).path
    if path not in ('/__resources__/session', '/resource-status.json') and not path.startswith('/__resources__/asset/'):
        return False
    if not _trusted(handler):
        _response(handler,403,{'error':'Only the Atlas loopback address is supported'})
        return True
    try:
        # Read-only live inventory. The old standalone manager linked to this
        # URL but returned a 404 because it had no matching GET handler.
        if path == '/resource-status.json':
            _response(handler,200,_inventory())
        elif path == '/__resources__/session':
            data=_inventory()
            _response(handler,200,{'token':handler.server.atlas_resources_token,
                                   'inventory':data,'network_on_read':False})
        else:
            # Permit only known pinned names. No user path, no directory listing,
            # no network, and do not expose the analyzed repository.
            name=path.removeprefix('/__resources__/asset/')
            if name not in GRAPHICS:
                _response(handler,404,{'error':'Unknown graphics asset'})
                return True
            from atlas_resources import verify, _resource_path
            verified, invalid, info=verify()
            if name not in verified or name in invalid:
                _response(handler,404,{'error':'Graphics asset not installed or corrupt'})
                return True
            sha=info['records'][name]['sha256']
            source=_resource_path(sha)
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 8_000_000:
                _response(handler,409,{'error':'Graphics cache file invalid'})
                return True
            body=source.read_bytes()
            if hashlib.sha256(body).hexdigest()!=sha:
                _response(handler,409,{'error':'Graphics asset hash mismatch'})
                return True
            mime='text/css; charset=utf-8' if name.endswith('.css') else 'text/javascript; charset=utf-8'
            handler.send_response(200)
            handler.send_header('Content-Type',mime)
            handler.send_header('Content-Length',str(len(body)))
            handler.send_header('Content-Disposition',f'attachment; filename="{name}"')
            handler.send_header('Cache-Control','no-store')
            handler.send_header('Cross-Origin-Resource-Policy','same-origin')
            handler.send_header('X-Content-Type-Options','nosniff')
            handler.end_headers()
            handler.wfile.write(body)
    except (OSError,ValueError,KeyError) as exc:
        _response(handler,500,{'error':str(exc)[:240]})
    return True


def post(handler) -> bool:
    path=urlsplit(handler.path).path
    if path not in ('/__resources__/download','/__resources__/import'):
        return False
    if not _trusted(handler,require_origin=True):
        _response(handler,403,{'error':'Atlas local origin and session token required'})
        return True
    content_length=handler.headers.get('Content-Length','')
    try:
        length=int(content_length)
    except ValueError:
        _response(handler,400,{'error':'Content-Length required'})
        return True
    maximum=512 if path.endswith('/download') else MAX_ARCHIVE
    if length<=0 or length>maximum:
        _response(handler,413,{'error':'Request is empty or over the configured size limit'})
        return True
    lock=handler.server.atlas_resources_lock
    if not lock.acquire(blocking=False):
        _response(handler,409,{'error':'Another installation is already running'})
        return True
    try:
        if path.endswith('/download'):
            if handler.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                raise ValueError('Expected JSON request')
            body=json.loads(handler.rfile.read(length))
            if not isinstance(body,dict) or set(body)!={'package'} or body['package'] not in PINNED_FILES:
                raise ValueError('Only pinned catalog package names are accepted')
            import atlas_resources
            result=download_pinned_package(atlas_resources,body['package'],allow_network=True)
        else:
            if handler.headers.get('Content-Type','').split(';')[0].strip()!='application/octet-stream':
                raise ValueError('Expected one .tgz binary file')
            if handler.headers.get('X-Atlas-Manual-Consent') != 'unverified-origin-accepted':
                raise ValueError('A manually supplied archive requires explicit consent about unverified origin')
            from atlas_resources import CACHE
            if CACHE.is_symlink():
                raise ValueError('Resource cache may not be symlinked')
            CACHE.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=CACHE,suffix='.tgz',delete=False) as file:
                tmp=Path(file.name)
                remaining=length
                while remaining:
                    block=handler.rfile.read(min(1024*1024,remaining))
                    if not block:
                        raise ValueError('Incomplete upload')
                    file.write(block)
                    remaining-=len(block)
            try:
                import atlas_resources
                result=import_npm(atlas_resources,tmp,accept_unverified_origin=True)
            finally:
                tmp.unlink(missing_ok=True)
        if all(package_ready(__import__('atlas_resources'),name) for name in PINNED_FILES):
            __import__('atlas_resources').activate_graphics()
            if handler.site is not None:
                _sync_site(handler.site)
        status = 200
        payload = {'ok':True,'result':result,'inventory':_inventory(),
                   'hint':'Refresh the graph after all required assets are installed.'}
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        status = 400
        payload = {'ok':False,'error':str(exc)[:400]}
    finally:
        # Release only after installation and final inventory are complete,
        # but BEFORE sending HTTP response. The client can otherwise receive
        # 200 and immediately get a spurious 409 on its next request.
        lock.release()
    _response(handler,status,payload)
    return True


class _StandaloneManager(BaseHTTPRequestHandler):
    """Only resource-manager HTML and explicit protected endpoints, no project."""
    site = None

    def do_GET(self):
        if get(self):
            return
        if urlsplit(self.path).path not in ('/', '/resource-manager.html'):
            self.send_error(404, 'Only the local resource manager is exposed')
            return
        from atlas_resource_catalog import html_report
        try:
            page = html_report(_inventory(), standalone=True).encode('utf8')
        except (OSError,ValueError) as exc:
            self.send_error(500, str(exc)[:200]);return
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(page)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if not post(self):
            self.send_error(404, 'Unknown resource action')


def serve_standalone(*, port: int = 0, open_browser: bool = True):
    """Local UI independent of repository selection; no socket until invoked."""
    with ThreadingHTTPServer(('127.0.0.1',port),_StandaloneManager) as server:
        configure_server(server)
        address=f'http://127.0.0.1:{server.server_address[1]}/resource-manager.html'
        print('Atlas offline graphics manager:',address,flush=True)
        print('No downloads until you click; Ctrl+C to exit.',flush=True)
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('Atlas resource manager stopped.',flush=True)
