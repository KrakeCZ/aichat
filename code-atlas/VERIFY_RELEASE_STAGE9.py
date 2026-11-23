#!/usr/bin/env python3
"""Verify every distributed file hash against the Stage 9 manifest."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int,list[str]]:
    root = root.resolve()
    manifest=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text('utf8'))
    errs=[]
    for relative,digest in manifest['files'].items():
        f=(root/relative).resolve()
        if not f.is_relative_to(root) or not f.is_file():
            errs.append('MISSING: '+relative)
        elif hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            errs.append('MISMATCH: '+relative)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'
            and p.name!='RELEASE_FILE_MANIFEST.json'}
    errs += ['UNLISTED: '+f for f in sorted(actual-set(manifest['files']))]
    return len(manifest['files']),errs

if __name__=='__main__':
    count,issues=verify(Path(__file__).parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage 9 manifest: {count} source files verified (SHA-256).')
