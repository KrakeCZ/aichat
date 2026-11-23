#!/usr/bin/env python3
"""Verify the immutable full Stage 29 source manifest without networking."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    n,problems=verify(Path(__file__).resolve().parent)
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Stage 29 full release: {n}/{n} SHA-256 files verified.')
