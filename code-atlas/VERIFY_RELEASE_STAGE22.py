#!/usr/bin/env python3
"""Verify distributed Stage 22 source and cached public graphics; no network."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    count, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 22: {count} files verified by SHA-256.')
