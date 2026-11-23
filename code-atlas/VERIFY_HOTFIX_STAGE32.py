#!/usr/bin/env python3
"""Verify only Stage 32 overlay files, without reading project output/cache."""
from __future__ import annotations
import hashlib,json
from pathlib import Path


def verify(root:Path) -> tuple[int,list[str]]:
    root=Path(root).resolve()
    manifest=json.loads((root/'HOTFIX_STAGE32_SHA256.json').read_text(encoding='utf-8'))
    failures=[]
    for name,expected in manifest['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            failures.append('MISSING_OR_OUTSIDE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            failures.append('SHA256_MISMATCH '+name)
    return len(manifest['files']),failures

if __name__=='__main__':
    checked,problems=verify(Path(__file__).resolve().parent)
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Stage 32 hotfix: {checked}/{checked} changed files SHA-256 verified.')
