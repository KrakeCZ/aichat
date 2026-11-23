#!/usr/bin/env python3
"""Verify complete Stage 21 distribution without network, based on its SHA-256 manifest."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    files, errors = verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage 21: {files} bundled files verified by SHA-256.')
