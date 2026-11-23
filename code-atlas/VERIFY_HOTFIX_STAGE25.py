#!/usr/bin/env python3
"""Verify the overlay files staged by the Stage 25 hotfix manifest."""
import hashlib
import json
from pathlib import Path

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    expected=json.loads((root/'STAGE25_HOTFIX_HASHES.json').read_text('utf-8'))
    problems=[]
    for rel,sha in expected.items():
        path=root/rel
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            problems.append(rel)
    if problems:raise SystemExit('Stage 25 hotfix files incorrect: '+', '.join(problems))
    print('Stage 25 hotfix:',len(expected),'SHA-256 records verified.')
