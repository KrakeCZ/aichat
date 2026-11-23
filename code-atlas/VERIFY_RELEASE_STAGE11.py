#!/usr/bin/env python3
"""Verify the immutable Stage 11 release manifest. No project code is executed."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text('utf-8'))
    errors=[]
    for name,sha in manifest['files'].items():
        file=(root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            errors.append(f'MISSING {name}')
        elif hashlib.sha256(file.read_bytes()).hexdigest()!=sha:
            errors.append(f'MISMATCH {name}')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
        and p.name!='RELEASE_FILE_MANIFEST.json' and '__pycache__' not in p.parts
        and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}
    errors.extend(f'UNLISTED {x}' for x in actual-set(manifest['files']))
    return len(manifest['files']), errors

if __name__=='__main__':
    count,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage 11 SHA-256: {count} files verified.')
