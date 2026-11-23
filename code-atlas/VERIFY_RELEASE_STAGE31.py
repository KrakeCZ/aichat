#!/usr/bin/env python3
"""Check integrity of the complete Stage31 offline distribution."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify


if __name__ == '__main__':
    count, failures = verify(Path(__file__).resolve().parent)
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Stage 31 full release: {count}/{count} SHA-256 files verified.')
