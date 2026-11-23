#!/usr/bin/env python3
"""Verify only explicit Stage41 overlay files; no scan of user projects."""
from __future__ import annotations
from pathlib import Path
import hashlib, json


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    manifest=json.loads((root/'HOTFIX_STAGE41_SHA256.json').read_text(encoding='utf8'))
    problems=[]
    for rel,expected in manifest['files'].items():
        path=root/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            problems.append('UNSAFE_OR_MISSING '+rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            problems.append('SHA256_MISMATCH '+rel)
    return len(manifest['files']),problems

if __name__=='__main__':
    total,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage41 hotfix: {total}/{total} SHA-256 verified. Project files not accessed.')
