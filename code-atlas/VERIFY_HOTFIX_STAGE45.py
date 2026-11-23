#!/usr/bin/env python3
"""Stage45 hotfix integrity: verify only explicitly shipped files, offline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    manifest=root/'HOTFIX_STAGE45_SHA256.json'
    data=json.loads(manifest.read_text(encoding='utf8'))
    hashes=data['files']
    errors=[]
    for name,expected in hashes.items():
        relative=Path(name)
        target=root/relative
        if (relative.is_absolute() or '..' in relative.parts or
            target.is_symlink() or not target.is_file() or
            not target.resolve().is_relative_to(root)):
            errors.append('MISSING_OR_UNSAFE: '+name)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            errors.append('SHA256_MISMATCH: '+name)
    return len(hashes),errors


if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage45 hotfix: {n}/{n} SHA-256 verified; project files not inspected.')
