#!/usr/bin/env python3
"""Optional byte-for-byte reference test against the user's original V0 ZIP.

The distributable Universal Atlas deliberately does NOT include the old app's
prepared graph, source index or portal in somebody else's project bundle.
Usage: python VERIFY_ORIGINAL_V0.py --reference /path/to/original.zip
Or:    python VERIFY_ORIGINAL_V0.py --reference /path/to/original-v0-directory
"""
import argparse
import hashlib
from pathlib import Path
import zipfile


def verify(reference: Path, manifest: Path) -> tuple[int, list[str]]:
    expected = {name:digest for digest,name in (line.split('  ', 1) for line in manifest.read_text().splitlines() if line.strip())}
    missing = []
    if reference.is_dir():
        for name, digest in expected.items():
            file = reference / name
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                missing.append(name)
    elif reference.is_file() and zipfile.is_zipfile(reference):
        with zipfile.ZipFile(reference) as z:
            names = z.namelist()
            for name, digest in expected.items():
                matches = [n for n in names if n.endswith('/aichat-portal-clean/' + name) or n == name]
                if len(matches) != 1 or hashlib.sha256(z.read(matches[0])).hexdigest() != digest:
                    missing.append(name)
    else:
        raise ValueError('Reference musí být existující adresář nebo původní ZIP.')
    return len(expected), missing


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Porovná původní V0 podle historických SHA-256; neobsahuje jeho zdrojáky.')
    ap.add_argument('--reference', type=Path, required=True, help='Původní ZIP nebo rozbalená složka aichat-portal-clean')
    args = ap.parse_args()
    count, missing = verify(args.reference, Path(__file__).with_name('ORIGINAL_V0_SHA256.txt'))
    if missing:
        print('V0 reference mismatch:', missing)
        raise SystemExit(1)
    print(f'ORIGINAL V0 OK: {count} souborů, SHA-256 odpovídá referenci.')
