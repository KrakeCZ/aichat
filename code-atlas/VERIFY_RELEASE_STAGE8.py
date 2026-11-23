#!/usr/bin/env python3
"""Stage 8 binary snapshot verification; no analysis or network access."""
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int, list[str]]:
    root = root.resolve()
    manifest = json.loads((root / 'RELEASE_FILE_MANIFEST.json').read_text(encoding='utf8'))
    errors = []
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append('MISSING: ' + relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('MISMATCH: ' + relative)
    actual = {x.relative_to(root).as_posix() for x in root.rglob('*')
              if x.is_file() and '__pycache__' not in x.parts and '.pytest_cache' not in x.parts
              and x.name != 'RELEASE_FILE_MANIFEST.json' and x.suffix != '.pyc'}
    if actual != set(manifest['files']):
        errors.extend('UNLISTED: ' + x for x in sorted(actual - set(manifest['files'])))
    return len(manifest['files']), errors

if __name__ == '__main__':
    count, errors = verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage 8: {count} souborů odpovídá manifestu SHA-256')
