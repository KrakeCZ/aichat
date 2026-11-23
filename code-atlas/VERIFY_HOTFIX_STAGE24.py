#!/usr/bin/env python3
"""Verify only Stage 24 overlay files; leave existing user data untouched."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root/'STAGE24_HOTFIX_HASHES.json').read_text('utf8'))
issues = []
for rel, expected in manifest['files'].items():
    candidate = (root/rel).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        issues.append(f'MISSING {rel}')
    elif hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
        issues.append(f'MISMATCH {rel}')
if issues:
    raise SystemExit('\n'.join(issues))
print(f'Stage 24 hotfix: {len(manifest["files"])} overlay files verified; user files untouched.')
