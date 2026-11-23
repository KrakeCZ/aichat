#!/usr/bin/env python3
"""Verify Stage 32 distribution with the bundled local SHA-256 manifest."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__=='__main__':
    count,failures=verify(Path(__file__).resolve().parent)
    # A user may place an existing generated Atlas (.code-atlas) beside
    # the program. It is private output, not source-release content.
    failures=[x for x in failures if not x.startswith('UNLISTED .code-atlas/')]
    if failures:raise SystemExit('\n'.join(failures))
    print(f'Stage 32 release: {count}/{count} files verified (SHA-256).')
