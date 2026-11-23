#!/usr/bin/env python3
"""Verify only Stage 27's changed files; leave user's installed resources alone."""
from pathlib import Path
import hashlib
import json


def verify(root: Path):
    root = root.resolve()
    info = json.loads((root/'HOTFIX_STAGE27_SHA256.json').read_text(encoding='utf-8'))
    issues = []
    for rel, expected in info['files'].items():
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            issues.append('MISSING_OR_SYMLINK '+rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            issues.append('MISMATCH '+rel)
    return len(info['files']), issues


if __name__ == '__main__':
    n, issues = verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage 27 hotfix: {n} changed files verified; existing offline libraries unchanged.')
