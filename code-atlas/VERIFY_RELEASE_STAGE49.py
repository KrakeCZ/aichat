#!/usr/bin/env python3
"""Verify the Stage49 source archive; additional local runtime data remain private."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify
if __name__=='__main__':
    count,issues=verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage49 source release: {count}/{count} SHA-256 verified.')
