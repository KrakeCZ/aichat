#!/usr/bin/env python3
"""Verify Stage37 complete source manifest without downloads or project execution."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    total,errors=verify(Path(__file__).resolve().parent)
    if errors: raise SystemExit('\n'.join(errors))
    print(f'Stage37 complete release: {total}/{total} SHA-256 verified.')
