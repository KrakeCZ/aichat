#!/usr/bin/env python3
"""Verify Stage31 changed files only, never traverse user repositories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int, list[str]]:
    root = Path(root).resolve()
    meta = json.loads((root/'HOTFIX_STAGE31_SHA256.json').read_text(encoding='utf-8'))
    errors: list[str] = []
    for relative, expected in meta['files'].items():
        target = root / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root) or not target.is_file():
            errors.append('MISSING_OR_OUTSIDE ' + relative)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            errors.append('SHA256_MISMATCH ' + relative)
    return len(meta['files']), errors


if __name__ == '__main__':
    count, failures = verify(Path(__file__).resolve().parent)
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Stage 31 hotfix: {count}/{count} SHA-256 files verified.')
