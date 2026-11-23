"""Portable, hash-checked offline resource packs. Zero network operations.

No zip extraction or dynamic code execution; all bytes are validated before
installation. Sources are explicitly selected Atlas cache or one user file.
"""
from __future__ import annotations
import hashlib
import json
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

SCHEMA = 'atlas-offline-pack/1'
MAX_ITEMS = 64
MAX_SINGLE_BYTES = 250_000_000
MAX_TOTAL_BYTES = 500_000_000
MAX_NOTICE_BYTES = 300_000
NOTICE_NAME = re.compile(r'[A-Za-z0-9_.@-]{1,100}\.LICENSE\.txt')


def _check(name: str, record: dict) -> None:
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',name):
        raise ValueError('Invalid resource ID')
    if not isinstance(record,dict) or record.get('category') not in ('graphics','grammar','toolchain'):
        raise ValueError('Invalid resource category')
    if not re.fullmatch(r'[0-9a-f]{64}', record.get('sha256','')):
        raise ValueError('Invalid SHA-256')
    if not isinstance(record.get('bytes'),int) or not 0<record['bytes']<=MAX_SINGLE_BYTES:
        raise ValueError('Invalid resource size')


def create_pack(manager, output: Path, selected: list[str] | None=None) -> dict:
    ok,bad,data=manager.verify()
    if bad:
        raise ValueError('Invalid cached files: '+', '.join(bad))
    names=sorted(set(selected)) if selected else sorted(ok)
    if not names:
        raise ValueError('No installed resources to export')
    if len(names)>MAX_ITEMS or any(n not in ok for n in names):
        raise ValueError('Selected resource not installed or too many resources')
    records={n:data['records'][n] for n in names}
    for n,r in records.items():_check(n,r)
    if sum(r['bytes'] for r in records.values())>MAX_TOTAL_BYTES:
        raise ValueError('Offline pack exceeds 500 MB')
    # Include only license notices in the resource manager's own fixed folder.
    # An absent notice remains visible as absent; no invented upstream license.
    notices={}
    license_dir=manager.CACHE/'licenses'
    if license_dir.is_dir() and not license_dir.is_symlink():
        for file in sorted(license_dir.iterdir()):
            if not NOTICE_NAME.fullmatch(file.name) or not file.is_file() or file.is_symlink():
                continue
            if file.stat().st_size>MAX_NOTICE_BYTES:
                raise ValueError('License notice is too large')
            content=file.read_bytes()
            if len(content)>0:
                notices[file.name]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_symlink():raise ValueError('Refusing symlink output')
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.tmp',delete=False) as tmp:
        tmp_path=Path(tmp.name)
    try:
        with ZipFile(tmp_path,'w',ZIP_DEFLATED,compresslevel=6) as z:
            z.writestr('resource-pack.json',json.dumps({'schema':SCHEMA,'resources':records,'license_notices':notices},sort_keys=True,indent=2)+'\n')
            for name,rec in records.items():
                blob=manager._resource_path(rec['sha256'])
                if blob.is_symlink() or not blob.is_file() or manager._hash(blob)!=rec['sha256']:
                    raise ValueError('Cache content invalid: '+name)
                z.write(blob,'blobs/'+rec['sha256'])
            for name in sorted(notices):
                z.write(license_dir/name,'licenses/'+name)
        tmp_path.replace(output)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {'file':str(output),'count':len(records),'size':output.stat().st_size,
            'sha256':manager._hash(output)}


def install_pack(manager, source: Path) -> dict:
    source=Path(source)
    if source.is_symlink() or not source.is_file() or source.stat().st_size>MAX_TOTAL_BYTES:
        raise ValueError('Invalid explicitly selected pack')
    try:
        with ZipFile(source,'r') as z:
            names=z.namelist()
            if not names or len(names)>MAX_ITEMS+1+MAX_ITEMS or len(set(names))!=len(names):
                raise ValueError('Malformed archive inventory')
            if 'resource-pack.json' not in names or z.getinfo('resource-pack.json').file_size>200_000:
                raise ValueError('Manifest missing or too large')
            data=json.loads(z.read('resource-pack.json'))
            if data.get('schema')!=SCHEMA or not isinstance(data.get('resources'),dict):
                raise ValueError('Unknown offline resource pack')
            records=data['resources']
            notices=data.get('license_notices',{})
            if not isinstance(notices,dict) or len(notices)>MAX_ITEMS:
                raise ValueError('Invalid license-notice count')
            for name,rec in notices.items():
                if not NOTICE_NAME.fullmatch(name) or not isinstance(rec,dict) or \
                   not isinstance(rec.get('bytes'),int) or not 0<rec['bytes']<=MAX_NOTICE_BYTES or \
                   not re.fullmatch(r'[0-9a-f]{64}',rec.get('sha256','')):
                    raise ValueError('Invalid license-notice inventory')
            if len(records)>MAX_ITEMS or len(records)==0:
                raise ValueError('Invalid resource count')
            for name,rec in records.items():_check(name,rec)
            blobs={r['sha256'] for r in records.values()}
            expected={'resource-pack.json'}|{'blobs/'+sha for sha in blobs}|{'licenses/'+name for name in notices}
            if set(names)!=expected:
                raise ValueError('Unexpected archive entries')
            for sha in blobs:
                info=z.getinfo('blobs/'+sha)
                expected_bytes=next(r['bytes'] for r in records.values() if r['sha256']==sha)
                if info.file_size!=expected_bytes or info.file_size>MAX_SINGLE_BYTES:
                    raise ValueError('Archive file size mismatch')
            for name,rec in notices.items():
                info=z.getinfo('licenses/'+name)
                if info.file_size!=rec['bytes']:
                    raise ValueError('License notice archive size mismatch')
            if sum(z.getinfo('blobs/'+sha).file_size for sha in blobs)>MAX_TOTAL_BYTES:
                raise ValueError('Archive expands beyond maximum allowed size')
            # Validate EVERY blob BEFORE importing anything.
            for sha in blobs:
                h=hashlib.sha256()
                with z.open('blobs/'+sha) as stream:
                    while chunk:=stream.read(1024*1024):h.update(chunk)
                if h.hexdigest()!=sha:
                    raise ValueError('Offline pack SHA-256 failed: '+sha)
            # License notices are checksummed and cannot refer to arbitrary paths.
            for name,rec in notices.items():
                h=hashlib.sha256()
                with z.open('licenses/'+name) as stream:
                    while chunk:=stream.read(1024*1024):h.update(chunk)
                if h.hexdigest()!=rec['sha256']:
                    raise ValueError('License notice SHA-256 mismatch')
            # Import only known records; never extract paths from the archive.
            with tempfile.TemporaryDirectory(prefix='atlas-pack-') as tmp:
                for sha in sorted(blobs):
                    f=Path(tmp)/sha
                    with z.open('blobs/'+sha) as inp, f.open('wb') as out:
                        while chunk:=inp.read(1024*1024):out.write(chunk)
                    for name,rec in sorted(records.items()):
                        if rec['sha256']==sha:
                            manager.import_resource(name=name,source=f,category=rec['category'],
                                expected_sha=sha,origin=str(rec.get('origin',''))[:500])
            if notices:
                notice_dir=manager.CACHE/'licenses'
                notice_dir.mkdir(parents=True,exist_ok=True)
                for name in sorted(notices):
                    with z.open('licenses/'+name) as source:
                        content=source.read(MAX_NOTICE_BYTES+1)
                    dest=notice_dir/name
                    if dest.is_symlink():
                        raise ValueError('Refusing symlink license notice path')
                    dest.write_bytes(content)
    except (BadZipFile,KeyError,TypeError,ValueError,json.JSONDecodeError) as e:
        raise ValueError('Resource pack was rejected: '+str(e)) from e
    return {'installed':sorted(records),'count':len(records),'network':'disabled'}
