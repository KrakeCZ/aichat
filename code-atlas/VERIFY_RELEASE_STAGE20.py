#!/usr/bin/env python3
"""Verify every distributed Stage 20 file against SHA-256, fully offline."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    count, errors = verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage 20: {count} source files verified against SHA-256 manifest; no network.')
