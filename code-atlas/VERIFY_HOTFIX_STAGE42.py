#!/usr/bin/env python3
"""Check explicit Stage 42 shipped overlay, no project traversal/network."""
from pathlib import Path
import hashlib,json

def verify(root:Path):
    root=root.resolve()
    expected=json.loads((root/'HOTFIX_STAGE42_SHA256.json').read_text())['files']
    errors=[]
    for name,digest in expected.items():
        p=root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file()
            or p.is_symlink() or not p.resolve().is_relative_to(root)):
            errors.append('UNSAFE_OR_MISSING '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            errors.append('SHA256_MISMATCH '+name)
    return len(expected),errors

if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage42 hotfix: {n}/{n} SHA-256 verified. Project data not accessed.')
