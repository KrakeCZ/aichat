#!/usr/bin/env python3
"""Offline full-source manifest verifier; do not traverse user repository."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify
if __name__=='__main__':
    count,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage46.2 complete release: {count}/{count} SHA-256 verified; private user caches excluded.')
