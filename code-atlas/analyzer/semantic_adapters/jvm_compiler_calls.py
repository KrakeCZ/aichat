"""Legacy JVM descriptor utility, retained for source-only compatibility.

The old module used to compile analyzed Kotlin/Java projects to generate
bytecode evidence. That behavior is forbidden by the Atlas static-only
contract. Nothing in this module launches a compiler or reads user files.
"""
from __future__ import annotations

def _arity(descriptor: str) -> int | None:
    """Return JVM method parameter count; malformed descriptors are rejected."""
    if not descriptor.startswith('(') or ')' not in descriptor:
        return None
    raw = descriptor[1:descriptor.index(')')]
    i = 0
    count = 0
    while i < len(raw):
        while i < len(raw) and raw[i] == '[':
            i += 1
        if i >= len(raw):
            return None
        if raw[i] == 'L':
            end = raw.find(';', i + 1)
            if end < 0:
                return None
            i = end + 1
        elif raw[i] in 'ZBCSIJFD':
            i += 1
        else:
            return None
        count += 1
    return count

