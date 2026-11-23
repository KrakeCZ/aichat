#!/usr/bin/env python3
"""Verify the immutable Stage 12 file manifest; does not execute project code."""
import hashlib, json
from pathlib import Path

def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'RELEASE_FILE_MANIFEST.json').read_text('utf-8'))
    errors = []
    for rel, expected in manifest['files'].items():
        file = (root / rel).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            errors.append(f'MISSING {rel}')
        elif hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            errors.append(f'MISMATCH {rel}')
    actual = {f.relative_to(root).as_posix() for f in root.rglob('*')
              if f.is_file() and f.name != 'RELEASE_FILE_MANIFEST.json'
              and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts
              and f.suffix != '.pyc'}
    errors.extend(f'UNLISTED {x}' for x in actual - set(manifest['files']))
    return len(manifest['files']), errors

if __name__ == '__main__':
    count, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 12 SHA-256: {count} files verified.')
