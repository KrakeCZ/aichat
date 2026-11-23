#!/usr/bin/env python3
"""Verify every distributed file against the Stage 16 checksum manifest."""
from pathlib import Path
from VERIFY_RELEASE_STAGE15 import verify

if __name__ == '__main__':
    total, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 16 release: {total} files checked against SHA-256 manifest.')
