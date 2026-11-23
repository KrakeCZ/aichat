#!/usr/bin/env python3
"""Verify only Stage42.1 hotfix files; never traverse the user's project."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    expected=json.loads((root/'HOTFIX_STAGE42_1_SHA256.json').read_text(encoding='utf8'))['files']
    issues=[]
    for name,digest in expected.items():
        rel=Path(name)
        file=root/rel
        if rel.is_absolute() or '..' in rel.parts or file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            issues.append('UNSAFE_OR_MISSING: '+name)
        elif hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            issues.append('SHA256_MISMATCH: '+name)
    return len(expected),issues


if __name__=='__main__':
    n,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage42.1 hotfix: {n}/{n} SHA-256 verified. Project files untouched.')
