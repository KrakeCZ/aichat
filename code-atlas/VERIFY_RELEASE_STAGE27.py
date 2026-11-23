#!/usr/bin/env python3
"""Verify Stage 27's entire distributed source release using local SHA-256."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    total, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 27 release: {total} files verified with SHA-256.')
