#!/usr/bin/env python3
"""Verify Stage43 hotfix source/document bytes only; read-only and offline."""
from pathlib import Path
import hashlib,json

def verify(root: Path):
    root=root.resolve()
    hashes=json.loads((root/'HOTFIX_STAGE43_SHA256.json').read_text('utf8'))['files']
    issues=[]
    for name,expected in hashes.items():
        relative=Path(name)
        file=root/relative
        if relative.is_absolute() or '..' in relative.parts or file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            issues.append('MISSING_OR_UNSAFE: '+name)
        elif hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            issues.append('SHA256_MISMATCH: '+name)
    return len(hashes),issues

if __name__ == '__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage43 hotfix: {n}/{n} SHA-256 verified.')
