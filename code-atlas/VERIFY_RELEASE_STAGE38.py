#!/usr/bin/env python3
"""Verify full Stage38 source manifest without networking or code execution."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage38 complete release: {n}/{n} SHA-256 files verified.')
