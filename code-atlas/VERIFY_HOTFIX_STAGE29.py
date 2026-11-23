#!/usr/bin/env python3
"""Verify only Stage 29 hotfix, without touching Atlas caches or projects."""
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int, list[str]]:
    root=Path(root).resolve()
    info=json.loads((root/'HOTFIX_STAGE29_SHA256.json').read_text(encoding='utf-8'))
    problems=[]
    for name,digest in info['files'].items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            problems.append('MISSING_OR_OUTSIDE '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            problems.append('WRONG_SHA256 '+name)
    return len(info['files']),problems

if __name__=='__main__':
    n,problems=verify(Path(__file__).resolve().parent)
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Stage 29 hotfix: {n}/{n} SHA-256 files verified.')
