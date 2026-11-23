#!/usr/bin/env python3
"""Check every distributed file (except the self-referential manifest)."""
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> tuple[int,list[str]]:
    manifest=json.loads((root/'RELEASE_FILE_MANIFEST.json').read_text(encoding='utf8'))
    errors=[]
    for name,expected in manifest.get('files',{}).items():
        item=(root/name).resolve()
        if not item.is_relative_to(root.resolve()) or not item.is_file():
            errors.append('MISSING: '+name)
        elif hashlib.sha256(item.read_bytes()).hexdigest()!=expected:
            errors.append('MISMATCH: '+name)
    return len(manifest.get('files',{})),errors

if __name__=='__main__':
    count,problems=verify(Path(__file__).resolve().parent)
    if problems:raise SystemExit('\n'.join(problems))
    print('Stage 7 SHA-256: '+str(count)+' souborů ověřeno')
