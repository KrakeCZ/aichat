#!/usr/bin/env python3
"""Verify Stage42.1.1 full source release manifest; no network/execution."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    n,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage 42.1.1 complete release: {n}/{n} SHA-256 verified.')
