#!/usr/bin/env python3
"""Read-only manifest verifier for Stage46.2 hotfix, not a user repo scanner."""
from pathlib import Path
import hashlib,json

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'HOTFIX_STAGE46_2_SHA256.json').read_text(encoding='utf-8'))
    issues=[]
    for name,digest in manifest['files'].items():
        rel=Path(name)
        p=root/rel
        if (rel.is_absolute() or '..' in rel.parts or p.is_symlink() or
                not p.is_file() or not p.resolve().is_relative_to(root)):
            issues.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            issues.append('MISMATCH '+name)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage46.2 hotfix: {len(manifest["files"])}/{len(manifest["files"])} SHA-256 verified.')
