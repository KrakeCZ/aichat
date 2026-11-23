#!/usr/bin/env python3
"""Verify Stage37 overlay without scanning or modifying analyzed projects."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def verify(root: Path) -> tuple[int, list[str]]:
    root=root.resolve()
    meta=json.loads((root/'HOTFIX_STAGE37_SHA256.json').read_text(encoding='utf8'))
    failures=[]
    for name,digest in meta['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            failures.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            failures.append('WRONG_SHA256 '+name)
    return len(meta['files']),failures

if __name__=='__main__':
    total,problems=verify(Path(__file__).resolve().parent)
    if problems: raise SystemExit('\n'.join(problems))
    print(f'Stage37 hotfix: {total}/{total} SHA-256 verified; user project data not accessed.')
