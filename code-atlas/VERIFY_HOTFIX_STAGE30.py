#!/usr/bin/env python3
"""Verify the Stage 30 overlay, without scanning a selected project."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int, list[str]]:
    root=Path(root).resolve()
    manifest=json.loads((root/'HOTFIX_STAGE30_SHA256.json').read_text(encoding='utf-8'))
    errors=[]
    for name,digest in manifest['files'].items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append('MISSING_OR_UNTRUSTED '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            errors.append('WRONG_SHA256 '+name)
    return len(manifest['files']),errors


if __name__=='__main__':
    total,problems=verify(Path(__file__).resolve().parent)
    if problems: raise SystemExit('\n'.join(problems))
    print(f'Stage 30 hotfix: {total}/{total} verified (SHA-256).')
