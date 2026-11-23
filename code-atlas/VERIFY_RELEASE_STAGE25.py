#!/usr/bin/env python3
"""Verify every Stage 25 source asset against SHA-256 release manifest."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    n,problems=verify(Path(__file__).resolve().parent)
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Stage 25: {n} files verified by SHA-256.')
