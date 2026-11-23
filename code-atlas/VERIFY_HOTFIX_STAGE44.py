#!/usr/bin/env python3
"""Stage44 explicit hotfix file integrity; offline and read only."""
from pathlib import Path
import hashlib
import json


def verify(root: Path) -> tuple[int,list[str]]:
    root=root.resolve()
    files=json.loads((root/'HOTFIX_STAGE44_SHA256.json').read_text(encoding='utf8'))['files']
    errors=[]
    for name,expected in files.items():
        rel=Path(name)
        path=root/rel
        if (rel.is_absolute() or '..' in rel.parts or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            errors.append('UNSAFE_OR_MISSING '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            errors.append('HASH_MISMATCH '+name)
    return len(files),errors

if __name__=='__main__':
    count,errors=verify(Path(__file__).resolve().parent)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Stage44 hotfix: {count}/{count} SHA-256 verified; user projects untouched.')
