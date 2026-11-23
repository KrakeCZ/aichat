"""Verify an offline Universal Code Atlas *source package* after extraction.

No network, plugins, project code, package installers, or source evaluation.
Checks SHA-256 of every distributed file except the self-referential manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MANIFEST = 'RELEASE_FILE_MANIFEST.json'


def verify(root: Path, *, strict_extras: bool = False) -> dict:
    root=root.resolve()
    data=json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    expected=data.get('files')
    if not isinstance(expected,dict):
        raise ValueError('manifest missing a file hash mapping')
    invalid=[]
    verified=0
    for name,sha in expected.items():
        # Treat manifest records as untrusted paths: never follow a traversal,
        # absolute path, junction, or symlink out of the release directory.
        if (not isinstance(name,str) or not name or '\\' in name or
                '\x00' in name or ':' in name or any(p in ('', '.', '..') for p in name.split('/'))):
            invalid.append({'path':str(name),'reason':'invalid_manifest_path'});continue
        path=root.joinpath(*name.split('/'))
        current=path
        linked=False
        while current != root:
            if current.is_symlink():linked=True;break
            current=current.parent
        if linked or not path.is_file():
            invalid.append({'path':name,'reason':'missing_or_symlink'});continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha:
            invalid.append({'path':name,'reason':'sha256_mismatch'})
        else:
            verified += 1
    bundled_problems=len(invalid)
    extras=0
    if strict_extras:
        allowed=set(expected)|{MANIFEST}
        for file in root.rglob('*'):
            if not file.is_file() and not file.is_symlink():
                continue
            rel=file.relative_to(root).as_posix()
            if (rel in allowed or '__pycache__' in file.parts or
                    '.pytest_cache' in file.parts):
                continue
            extras += 1
            invalid.append({'path':rel,'reason':'unexpected_release_file'})
    return {'schema':'atlas-full-source-release-check/1',
            'valid':not invalid,
            'verified_files':verified,
            'bundled_files_valid':bundled_problems==0,
            'unexpected_files_count':extras,
            'expected_files':len(expected),
            'problems':invalid[:100],
            'manifest_count_matches':data.get('file_count')==len(expected),
            'source_package_only':True,
            'strict_extras_requested':strict_extras,
            'extra_files_remediation':('Run without --strict-extras to verify distributed members while keeping local JARs, reports and output data; use a fresh extracted release for pristine strict verification.' if extras else None),
            'note':'Only bundled file integrity; does not certify parser semantics or operating-system E2E.'}


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description='Offline SHA-256 verifier of a full source release')
    parser.add_argument('source_dir',nargs='?',default='.')
    parser.add_argument('--strict-extras',action='store_true',help='Fail if extra files appear in the source release')
    args=parser.parse_args()
    try:result=verify(Path(args.source_dir), strict_extras=args.strict_extras)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        print('VERIFY FAIL:',type(exc).__name__,str(exc)[:180],file=sys.stderr)
        return 2
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['valid'] and result['manifest_count_matches'] else 1


if __name__=='__main__':
    raise SystemExit(main())
