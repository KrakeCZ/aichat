"""Pinned Java 21 Unicode codepoint names for regex ``\\N{...}``.

Registry generated at *release building QA* from every
``java.lang.Character.getName(cp)`` that round-trips through
``Character.codePointOf(name)`` for U+0000..U+10FFFF. The app loads only
an immutable compressed data file; no JVM, subprocess, dynamic eval, file
write, or network access occurs during project source analysis.

This proves Unicode **names** for the pinned JDK21 profile. It does not
certify arbitrary Java Pattern grammar, compiler reader contexts, or other
JDK/Unicode revisions.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import zlib

_DATA_FILE = 'clojure_java21_unicode_character_names.txt.zlib'
_META_FILE = 'clojure_java21_unicode_character_names_provenance.json'

@lru_cache(maxsize=1)
def jdk21_name_registry_provenance() -> dict:
    data = json.loads(Path(__file__).with_name(_META_FILE).read_text(encoding='utf-8'))
    return data


@lru_cache(maxsize=1)
def _names() -> frozenset[bytes]:
    compressed = Path(__file__).with_name(_DATA_FILE).read_bytes()
    metadata = jdk21_name_registry_provenance()
    # Verify both immutable compressed artifact and its decoded source names.
    # This is one-time integrity checking, not a per-file project requirement.
    if hashlib.sha256(compressed).hexdigest() != metadata['compressed_sha256']:
        raise ValueError('corrupt_static_java21_unicode_character_name_registry')
    decompressed = zlib.decompress(compressed)
    if hashlib.sha256(decompressed).hexdigest() != metadata['uncompressed_sha256']:
        raise ValueError('corrupt_static_java21_unicode_character_name_registry')
    names=frozenset(decompressed.splitlines())
    if len(names)!=metadata['count']:
        raise ValueError('invalid_static_java21_unicode_character_name_count')
    return names


def jdk21_character_name_valid(name: str) -> bool:
    """True exactly for Java 21 ``Character.codePointOf`` public names.

    No Unicode normalization, trimming, underscore/hyphen replacement, or
    Python ``unicodedata.lookup`` is applied. Character names are ASCII and
    case-insensitive for this Java profile.
    """
    if not name or len(name)>256 or not name.isascii():
        return False
    return name.upper().encode('ascii') in _names()
