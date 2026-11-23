#!/usr/bin/env python3
"""Read-only verification of the Stage 33 overlay; never scans project data."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int, list[str]]:
    root = Path(root).resolve()
    manifest = json.loads((root / 'HOTFIX_STAGE33_SHA256.json').read_text(encoding='utf-8'))
    failures = []
    for rel, digest in manifest['files'].items():
        p = root / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            failures.append('MISSING_OR_OUTSIDE ' + rel)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            failures.append('SHA256_MISMATCH ' + rel)
    return len(manifest['files']), failures

if __name__ == '__main__':
    n, issues = verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage 33 hotfix: {n}/{n} files verified (SHA-256).')
