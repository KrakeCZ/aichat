#!/usr/bin/env python3
"""Offline verify complete Stage46.1 release without reading private caches."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify
if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage46.1 full release: {n}/{n} SHA-256 verified.')
