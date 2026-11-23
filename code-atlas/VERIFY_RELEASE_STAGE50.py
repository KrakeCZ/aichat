#!/usr/bin/env python3
"""Verify Stage50 distributed files; retain private runtime caches."""
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify
if __name__ == '__main__':
    count, issues = verify(Path(__file__).resolve().parent)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage50 source release: {count}/{count} SHA-256 verified.')
