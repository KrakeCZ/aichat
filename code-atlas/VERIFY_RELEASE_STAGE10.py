#!/usr/bin/env python3
"""Verify SHA-256 of every distributed Stage 10 source file."""
from pathlib import Path
import hashlib,json

def verify(root: Path):
    root=root.resolve()
    info=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text('utf8'))
    issues=[]
    for relative,digest in info['files'].items():
        f=(root/relative).resolve()
        if not f.is_relative_to(root) or not f.is_file():issues.append('MISSING '+relative)
        elif hashlib.sha256(f.read_bytes()).hexdigest()!=digest:issues.append('MISMATCH '+relative)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and
      not any(x in p.parts for x in ('__pycache__','.pytest_cache')) and
      p.suffix!='.pyc' and p.name!='RELEASE_FILE_MANIFEST.json'}
    issues.extend('UNLISTED '+p for p in sorted(actual-set(info['files'])))
    return len(info['files']),issues
if __name__=='__main__':
    count,issues=verify(Path(__file__).parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage 10 verified: {count} SHA-256 file hashes')
