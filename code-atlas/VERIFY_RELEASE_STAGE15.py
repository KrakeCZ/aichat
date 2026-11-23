#!/usr/bin/env python3
"""Validate Stage 15 archived files without invoking analyzed projects."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'RELEASE_FILE_MANIFEST.json').read_text('utf8'))
    problems = []
    for rel, expected in manifest['files'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append('MISSING ' + rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append('MISMATCH ' + rel)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.name != 'RELEASE_FILE_MANIFEST.json'
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
              and p.suffix != '.pyc'}
    problems.extend('UNLISTED ' + x for x in actual - set(manifest['files']))
    problems.extend('MISSING_IN_ARCHIVE ' + x for x in set(manifest['files']) - actual)
    return len(manifest['files']), problems

if __name__ == '__main__':
    n, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 15 SHA-256: {n} files verified.')
