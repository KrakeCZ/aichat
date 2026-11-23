#!/usr/bin/env python3
"""Legacy verifier: checks the current distribution against its manifest."""
from pathlib import Path
import hashlib
import json


def verify(base: Path) -> tuple[int,list[str]]:
    manifest=json.loads((base/'RELEASE_FILE_MANIFEST.json').read_text(encoding='utf-8'))
    errors=[]
    for name,digest in manifest['files'].items():
        path=(base/name).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            errors.append('MISSING: '+name)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            errors.append('MISMATCH: '+name)
    return len(manifest['files']),errors


if __name__=='__main__':
    base=Path(__file__).resolve().parent
    count,problems=verify(base)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'OK: SHA-256 odpovídá u {count} souborů dle aktuálního manifestu.')
