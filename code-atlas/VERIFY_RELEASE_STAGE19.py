#!/usr/bin/env python3
"""Stage 19 release integrity, entirely offline and read-only."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    files, problems = verify(Path(__file__).resolve().parent)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage 19: {files} SHA-256 hashes verified, no network.')
