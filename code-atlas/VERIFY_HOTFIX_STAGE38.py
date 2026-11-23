#!/usr/bin/env python3
"""Verify Stage38 overlay with SHA-256, without scanning user projects."""
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    meta=json.loads((root/'HOTFIX_STAGE38_SHA256.json').read_text(encoding='utf8'))
    errors=[]
    for name,digest in meta['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            errors.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            errors.append('WRONG_SHA256 '+name)
    return len(meta['files']),errors

if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage38 hotfix: {n}/{n} SHA-256 verified. User projects not accessed.')
