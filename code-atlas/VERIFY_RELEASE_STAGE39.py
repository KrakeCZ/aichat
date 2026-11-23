#!/usr/bin/env python3
"""Verify complete Stage39 archive manifest without project, runtime or network."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify
if __name__=='__main__':
    n, issues=verify(Path(__file__).resolve().parent)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage39 complete release: {n}/{n} SHA-256 files verified.')
