#!/usr/bin/env python3
"""Hash-check only Stage51 updated files, preserving local JARs and generated data."""
import hashlib
import json
from pathlib import Path
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    files=json.loads((root/'HOTFIX_STAGE51_SHA256.json').read_text(encoding='utf-8'))['files']
    problems=[]
    for name,digest in files.items():
        p=root/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            problems.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            problems.append('MISMATCH '+name)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage51 hotfix: {len(files)}/{len(files)} SHA-256 verified.')
