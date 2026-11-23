#!/usr/bin/env python3
"""Stage44 full-manifest verifier; respects Stage43 private-cache exceptions."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify

if __name__ == '__main__':
    total,errors=verify(Path(__file__).resolve().parent)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Stage44 complete release: {total}/{total} SHA-256 verified, private caches not read.')
