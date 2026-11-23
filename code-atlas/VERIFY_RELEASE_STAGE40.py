#!/usr/bin/env python3
"""Verify release manifest; does not execute user code or fetch dependencies."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    n,issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage40 source release: {n}/{n} SHA-256 files verified.')
