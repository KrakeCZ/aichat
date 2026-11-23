#!/usr/bin/env python3
"""Check distributed Stage51 files; local extras require an explicit flag."""
import argparse
from pathlib import Path
from VERIFY_RELEASE_STAGE43 import verify as _verify


def verify(root, *, allow_local_extras=False):
    count,issues=_verify(root)
    extras=[issue for issue in issues if issue.startswith('UNLISTED ')]
    failures=[issue for issue in issues if not (allow_local_extras and issue.startswith('UNLISTED '))]
    return count,failures,len(extras)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-local-extras',action='store_true',
                        help='Check every distributed file, allow additional unverified local files')
    args=parser.parse_args()
    count,issues,extras=verify(Path(__file__).resolve().parent,allow_local_extras=args.allow_local_extras)
    if issues:raise SystemExit('\n'.join(issues))
    print(f'Stage51: {count}/{count} distributed files SHA-256 verified.')
    if args.allow_local_extras:
        print(f'Additional local files: {extras}; their contents are not verified.')


if __name__=='__main__':main()
