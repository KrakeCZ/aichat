#!/usr/bin/env python3
"""SHA-256 manifest validation of the Stage 23 source and graphics archive."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    number, errors = verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage 23: {number} files verified by SHA-256.')
