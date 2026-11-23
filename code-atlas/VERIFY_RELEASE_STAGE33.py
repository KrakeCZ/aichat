#!/usr/bin/env python3
"""Read-only verification of the Stage 33 full offline release manifest."""
from pathlib import Path
from VERIFY_RELEASE_STAGE17 import verify

if __name__ == '__main__':
    count, issues = verify(Path(__file__).resolve().parent)
    issues = [i for i in issues if not i.startswith('UNLISTED .code-atlas/')]
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'Stage 33 full release: {count}/{count} files verified (SHA-256).')
