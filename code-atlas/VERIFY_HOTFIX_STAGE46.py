#!/usr/bin/env python3
"""Offline Stage46 overlay integrity check; do not inspect user-project data."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int,list[str]]:
    root = root.resolve()
    manifest = json.loads((root/'HOTFIX_STAGE46_SHA256.json').read_text(encoding='utf8'))
    problems=[]
    for name,digest in manifest['files'].items():
        rel=Path(name)
        path=root/rel
        if (rel.is_absolute() or '..' in rel.parts or
                not path.is_file() or path.is_symlink() or
                not path.resolve().is_relative_to(root)):
            problems.append('UNSAFE_OR_MISSING '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            problems.append('SHA256_MISMATCH '+name)
    return len(manifest['files']),problems


if __name__=='__main__':
    number,problems=verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage46 hotfix: {number}/{number} SHA-256 verified. No project data accessed.')
