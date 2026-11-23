#!/usr/bin/env python3
"""Verify full release content; no external programs or project code."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify
if __name__=='__main__':
    n,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage41 full source: {n}/{n} SHA-256 files verified.')
