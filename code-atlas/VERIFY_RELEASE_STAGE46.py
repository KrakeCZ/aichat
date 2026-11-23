#!/usr/bin/env python3
"""Verify Stage46 source hashes while respecting private local asset caches."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify


if __name__=='__main__':
    number,problems=verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage46 full release: {number}/{number} SHA-256 verified; private caches excluded.')
