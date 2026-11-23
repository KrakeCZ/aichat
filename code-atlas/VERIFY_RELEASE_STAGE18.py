#!/usr/bin/env python3
"""Stage 18 release integrity verifier, offline, read-only."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    n, errors = verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage 18: {n} SHA-256 checksums verified, zero network operations.')
