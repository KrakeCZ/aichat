#!/usr/bin/env python3
"""Stage43 offline release integrity check that respects explicitly private caches.

Only known manifest paths are hashed. Additional files in explicitly private
project indexes and local resource staging directories are never opened or
hashed. Unexpected *program* files elsewhere remain a release integrity error.
This verifier does not certify contents or provenance of user-supplied files.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

_PRIVATE_UNLISTED_PREFIXES = (
    '.code-atlas/',
    'clojure-jars/',
    'online-resources/inbox/',
    'online-resources/blobs/',
)


def verify(root: Path) -> tuple[int, list[str]]:
    root = root.resolve()
    manifest=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text(encoding='utf-8'))
    expected=manifest['files']
    problems=[]
    for name, digest in sorted(expected.items()):
        relative=Path(name)
        target=root/relative
        if (not name or relative.is_absolute() or '..' in relative.parts or
            target.is_symlink() or not target.is_file() or
            not target.resolve().is_relative_to(root)):
            problems.append('MISSING_OR_UNSAFE '+name)
        elif not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
            problems.append('INVALID_SHA256 '+name)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            problems.append('MISMATCH '+name)

    for file in root.rglob('*'):
        if not file.is_file() or file.name=='RELEASE_FILE_MANIFEST.json':
            continue
        rel=file.relative_to(root)
        if ('__pycache__' in rel.parts or '.pytest_cache' in rel.parts or
            rel.suffix=='.pyc'):
            continue
        name=rel.as_posix()
        if name in expected:
            continue
        if name.startswith(_PRIVATE_UNLISTED_PREFIXES):
            # Local private data are not part of the distributed release.
            # No contents of these files are inspected by this verifier.
            continue
        problems.append('UNLISTED '+name)
    return len(expected),problems


if __name__ == '__main__':
    count,issues=verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage43 complete release: {count}/{count} SHA-256 verified; user caches excluded.')
