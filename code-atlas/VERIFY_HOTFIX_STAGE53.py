#!/usr/bin/env python3
"""Read-only Stage53 patch SHA-256 check without scanning private project files."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'HOTFIX_STAGE53_SHA256.json').read_text(encoding='utf8'))
    problems=[]
    for name,digest in manifest['files'].items():
        relative=Path(name)
        target=root/relative
        if (not name or relative.is_absolute() or '..' in relative.parts or
                target.is_symlink() or not target.is_file() or
                not target.resolve().is_relative_to(root)):
            problems.append('MISSING_OR_UNSAFE '+name)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            problems.append('MISMATCH '+name)
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'Stage53 hotfix: {len(manifest["files"])}/{len(manifest["files"])} SHA-256 verified.')


if __name__=='__main__':main()
