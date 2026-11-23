#!/usr/bin/env python3
"""Stage39: verify only shipped overlay files; never visit user source projects."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    hashes=json.loads((root/'HOTFIX_STAGE39_SHA256.json').read_text(encoding='utf8'))['files']
    errors=[]
    for name,digest in hashes.items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append('MISSING_OR_UNSAFE: '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            errors.append('WRONG_SHA256: '+name)
    return len(hashes),errors


if __name__=='__main__':
    count,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage39 hotfix: {count}/{count} SHA-256 verified. Project data untouched.')
