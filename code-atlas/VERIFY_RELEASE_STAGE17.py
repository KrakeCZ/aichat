#!/usr/bin/env python3
"""Verify all files in the Stage 17 foundation release; no network."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int, list[str]]:
    root = root.resolve()
    manifest=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text(encoding='utf-8'))
    problems=[]
    expected=manifest.get('files',{})
    for name, sha in expected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append('MISSING '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            problems.append('MISMATCH '+name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
       and p.name!='RELEASE_FILE_MANIFEST.json' and '__pycache__' not in p.parts
       and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}
    problems.extend('UNLISTED '+p for p in sorted(actual-set(expected)))
    return len(expected), problems


if __name__=='__main__':
    n,issues=verify(Path(__file__).parent)
    if issues: raise SystemExit('\n'.join(issues))
    print(f'Atlas Stage 17 Foundation: {n} SHA-256 hashes verified.')
