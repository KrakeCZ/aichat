#!/usr/bin/env python3
"""SHA-256 verification for the full Stage 24 release."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    number, issues = verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage 24: {number} files verified by SHA-256.')
