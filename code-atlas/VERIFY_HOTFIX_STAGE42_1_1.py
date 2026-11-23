#!/usr/bin/env python3
"""Verify Stage42.1.1 hotfix; never scan the user's project."""
from pathlib import Path
import hashlib
import json

def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    metadata=json.loads((root/'HOTFIX_STAGE42_1_1_SHA256.json').read_text(encoding='utf8'))
    hashes=metadata['files']
    issues=[]
    for name,digest in hashes.items():
        rel=Path(name)
        file=root/rel
        if rel.is_absolute() or '..' in rel.parts or file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            issues.append('MISSING_OR_UNSAFE: '+name)
        elif hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            issues.append('SHA256_MISMATCH: '+name)
    return len(hashes),issues

if __name__ == '__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage 42.1.1 hotfix: {n}/{n} SHA-256 verified. Project files untouched.')
