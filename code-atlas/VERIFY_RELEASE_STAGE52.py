#!/usr/bin/env python3
"""Read-only SHA-256 verification of the Stage52 source distribution."""
import argparse
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--allow-local-extras',action='store_true',
                     help='For existing Windows Stage51 directories with separately owned tools/JARs.')
    args=cli.parse_args()
    count,issues=verify(Path(__file__).resolve().parent)
    extras=[issue for issue in issues if issue.startswith('UNLISTED ')]
    problems=[issue for issue in issues if not (args.allow_local_extras and issue.startswith('UNLISTED '))]
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage52: {count}/{count} SHA-256 verified.')
    if args.allow_local_extras:
        print(f'Unlisted local files (contents not verified): {len(extras)}')


if __name__=='__main__':main()
