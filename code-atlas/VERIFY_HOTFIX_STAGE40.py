#!/usr/bin/env python3
"""Check shipped Stage40 overlay hashes. No project scan, network, or run."""
from pathlib import Path
import hashlib,json


def verify(base: Path):
    base=base.resolve()
    data=json.loads((base/'HOTFIX_STAGE40_SHA256.json').read_text(encoding='utf8'))
    failures=[]
    for rel, expected in data['files'].items():
        path=base/rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base):
            failures.append('UNSAFE_OR_MISSING '+rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append('SHA256_MISMATCH '+rel)
    return len(data['files']),failures


if __name__=='__main__':
    n,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage40 hotfix: {n}/{n} SHA-256 verified. No project data accessed.')
