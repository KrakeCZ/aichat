#!/usr/bin/env python3
"""Verify all distributed files of the Stage 26 source release."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    count, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 26: {count} source files verified against the SHA-256 manifest.')
