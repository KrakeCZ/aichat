"""Conservative JS/TS lexical scanner for import specifiers.

Never searches comments, ordinary strings, regex contents or template literals for
'import' tokens. This is not a full JS/TS parser: unsupported constructs are
omitted rather than represented as high-confidence dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass
from bisect import bisect_right
import re

@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    start: int


def tokens(source: str) -> list[Token]:
    result: list[Token] = []
    n = len(source)
    i = 0
    while i < n:
        char = source[i]
        if char.isspace():
            i += 1
            continue
        if char == '/' and i + 1 < n and source[i+1] == '/':
            end = source.find('\n', i+2)
            i = n if end < 0 else end
            continue
        if char == '/' and i + 1 < n and source[i+1] == '*':
            end = source.find('*/', i+2)
            i = n if end < 0 else end+2
            continue
        if char in ('"', "'", '`'):
            start, quote = i, char
            i += 1
            value = []
            while i < n:
                ch = source[i]
                if ch == '\\' and i + 1 < n:
                    value.append(source[i + 1])
                    i += 2
                    continue
                if ch == quote:
                    i += 1
                    break
                value.append(ch)
                i += 1
            # Template interpolation is deliberately not parsed.
            result.append(Token('template' if quote == '`' else 'string', ''.join(value), start))
            continue
        if char.isalpha() or char in '_$':
            start = i
            i += 1
            while i < n and (source[i].isalnum() or source[i] in '_$'):
                i += 1
            result.append(Token('word', source[start:i], start))
            continue
        # regex literal: heuristic skip if we are in an expression beginning,
        # e.g. /=, (, [, =, :, ! or 'return'. Ambiguous regex syntax is not
        # interpreted as an import declaration.
        if char == '/' and (not result or result[-1].value in ('=', '(', '[', ':', '!', 'return', '=>', ',')):
            start = i
            i += 1
            in_class = False
            while i < n and source[i] != '\n':
                if source[i] == '\\':
                    i += 2; continue
                if source[i] == '[': in_class = True
                elif source[i] == ']': in_class = False
                elif source[i] == '/' and not in_class:
                    i += 1
                    while i < n and source[i].isalpha(): i += 1
                    break
                i += 1
            result.append(Token('regex', source[start:i], start))
            continue
        result.append(Token('punct', char, i))
        i += 1
    return result


def imports(source: str) -> list[tuple[str, int]]:
    """Yield (module_specifier, one_based_line) for supported forms."""
    found: list[tuple[str, int]] = []
    ts = tokens(source)
    line_starts = [-1] + [i for i, c in enumerate(source) if c == '\n']
    for i, tok in enumerate(ts):
        if tok.kind != 'word' or tok.value not in ('import', 'export', 'require'):
            continue
        # Property access (object.import) and declarations are not module edges.
        if i and ts[i-1].value in ('.', '?.'):
            continue
        nxt = ts[i+1] if i+1 < len(ts) else None
        if not nxt:
            continue
        reference = None
        if tok.value == 'require':
            if nxt.value == '(' and i + 3 < len(ts) and ts[i+2].kind == 'string' and ts[i+3].value == ')':
                reference = ts[i+2].value
        elif tok.value == 'import' and nxt.kind == 'string':
            reference = nxt.value  # import './side-effect'
        elif tok.value == 'import' and nxt.value == '(':
            if i + 3 < len(ts) and ts[i+2].kind == 'string' and ts[i+3].value == ')':
                reference = ts[i+2].value  # import('./module')
        elif nxt.value not in ('.', ':', '=', '('):
            # Scan a bounded statement for from '...'; don't cross into a new declaration.
            for j in range(i+1, min(i+85, len(ts)-1)):
                here = ts[j]
                if here.value == ';' or (here.kind == 'word' and here.value in ('import', 'export')):
                    break
                if here.kind == 'word' and here.value == 'from' and ts[j+1].kind == 'string':
                    reference = ts[j+1].value
                    break
        if reference:
            line = bisect_right(line_starts, tok.start)
            found.append((reference, line))
    return found
