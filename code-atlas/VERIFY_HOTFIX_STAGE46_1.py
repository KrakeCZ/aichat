#!/usr/bin/env python3
"""Offline verify Stage46.1 overlay manifest only, no project file traversal."""
import hashlib,json
from pathlib import Path
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'HOTFIX_STAGE46_1_SHA256.json').read_text(encoding='utf8'))
    errors=[]
    for name,digest in manifest['files'].items():
        rel=Path(name);p=root/rel
        if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            errors.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            errors.append('SHA256_MISMATCH '+name)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage46.1 hotfix: {len(manifest["files"])}/{len(manifest["files"])} SHA-256 verified.')
