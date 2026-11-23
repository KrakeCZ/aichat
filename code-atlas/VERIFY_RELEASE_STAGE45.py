#!/usr/bin/env python3
"""Stage45 complete release verifier, delegates to private-cache-safe Stage43 verifier."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify

if __name__=='__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage45 complete release: {n}/{n} SHA-256 verified; private caches excluded.')
