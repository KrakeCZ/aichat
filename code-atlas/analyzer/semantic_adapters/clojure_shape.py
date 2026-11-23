"""Canonical *subset* of Clojure reader data, for offline differential evidence.

No source evaluation, macroexpansion, runtime imports, data readers or project
execution. This is a comparison of a deliberately restricted form-value subset,
NOT an AST or a complete Clojure reader/semantic resolver. Unsupported forms
raise UnsupportedShape; they are NEVER silently certified as equal.
"""
from __future__ import annotations

import re
from fractions import Fraction


class UnsupportedShape(ValueError):
    pass


_INTEGER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:N)?\Z")
_RATIO = re.compile(r"([+-]?(?:0|[1-9][0-9]*))/([1-9][0-9]*)\Z")
_HEX = re.compile(r"([+-]?)0[xX]([0-9a-fA-F]+)(N)?\Z")
_OCT = re.compile(r"([+-]?)0([0-7]+)(N)?\Z")
_RADIX = re.compile(r"([+-]?)([0-9]{1,2})[rR]([0-9A-Za-z]+)(N)?\Z")


def _atom(tag: str, value: str) -> str:
    return '@' + tag + str(len(value.encode('utf-8'))) + ':' + value


def _collection(tag: str, children: list[str]) -> str:
    return '[' + tag + str(len(children)) + ':' + ''.join(children) + ']'


def _decode_string(source: str) -> str:
    """Decode only JVM Clojure 1.12 escape forms already lexically validated."""
    out: list[str] = []
    i = 0
    escapes = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f', '"':'"', '\\':'\\'}
    while i < len(source):
        if source[i] != '\\':
            out.append(source[i]); i += 1; continue
        i += 1
        if i >= len(source):
            raise UnsupportedShape('truncated string escape')
        c = source[i]
        if c in escapes:
            out.append(escapes[c]); i += 1; continue
        if c == 'u':
            digits = source[i+1:i+5]
            if len(digits) != 4 or any(d not in '0123456789abcdefABCDEF' for d in digits):
                raise UnsupportedShape('unknown unicode escape')
            code = int(digits, 16)
            if 0xD800 <= code <= 0xDFFF:
                raise UnsupportedShape('surrogate string escape')
            out.append(chr(code)); i += 5; continue
        if c in '01234567':
            j = i
            while j < min(len(source), i+3) and source[j] in '01234567':
                j += 1
            code = int(source[i:j], 8)
            if code > 255:
                raise UnsupportedShape('out-of-range octal escape')
            out.append(chr(code)); i = j; continue
        raise UnsupportedShape('unknown string escape')
    return ''.join(out)


def _char_value(raw: str) -> str:
    if not raw.startswith('\\'):
        raise UnsupportedShape('not a Clojure char')
    text = raw[1:]
    named = {'newline':'\n', 'space':' ', 'tab':'\t', 'backspace':'\b',
             'formfeed':'\f', 'return':'\r'}
    if text in named:
        return named[text]
    if len(text) == 1:
        return text
    if text.startswith('u') and len(text) == 5:
        code = int(text[1:],16)
        if 0xD800 <= code <= 0xDFFF:
            raise UnsupportedShape('surrogate character')
        return chr(code)
    if text.startswith('o') and 1 < len(text) <= 4:
        code = int(text[1:],8)
        if code > 255:
            raise UnsupportedShape('octal character out of range')
        return chr(code)
    raise UnsupportedShape('unknown character')


def _number(raw: str) -> str | None:
    """Exact integer equivalents documented by the Clojure 1.12 reader.

    This is a bounded *value-identity* subset, NOT a complete number reader.
    Unrecognized numeric spellings must not silently become symbol evidence.
    """
    if len(raw) > 4096:
        raise UnsupportedShape('numeric token beyond bounded comparator')
    if _INTEGER.fullmatch(raw):
        return str(int(raw.rstrip('N')))
    m = _HEX.fullmatch(raw)
    if m:
        return str((-1 if m.group(1) == '-' else 1) * int(m.group(2), 16))
    m = _OCT.fullmatch(raw)
    if m:
        return str((-1 if m.group(1) == '-' else 1) * int(m.group(2), 8))
    m = _RADIX.fullmatch(raw)
    if m:
        base = int(m.group(2))
        if not 2 <= base <= 36:
            raise UnsupportedShape('invalid numeric radix')
        try:
            value = int(m.group(3), base)
        except ValueError as exc:
            raise UnsupportedShape('invalid numeric digit for radix') from exc
        return str((-1 if m.group(1) == '-' else 1) * value)
    m = _RATIO.fullmatch(raw)
    if m:
        denominator = int(m.group(2))
        if denominator == 0:
            raise UnsupportedShape('zero denominator')
        fraction = Fraction(int(m.group(1)), denominator)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f'{fraction.numerator}/{fraction.denominator}'
    return None


def canonical_form(form) -> str:
    """A type-distinguishing, length-prefixed Clojure *data* fingerprint.

    Maps and sets are sorted by canonical member value (the JVM reader does
    not preserve literal hash-map/set order). List/vector order is preserved.
    """
    kind = form.kind
    if kind == 'atom':
        value = form.value
        if value == 'nil':
            return _atom('0', '')
        if value in ('true', 'false'):
            return _atom('b', value)
        if value.startswith(':'):
            if value.startswith('::'):
                raise UnsupportedShape('auto-resolved keyword needs namespace context')
            return _atom('k', value)
        number = _number(value)
        if number is not None:
            return _atom('n', number)
        # Other numeric syntaxes (float, hex, scientific, BigDecimal...) are
        # not certified by this narrow comparator. Never label them symbols.
        if re.match(r'[+-]?(?:\d|\d*\.\d)', value):
            raise UnsupportedShape('numeric format unsupported by comparator')
        return _atom('s', value)
    if kind == 'string':
        return _atom('t', _decode_string(form.value))
    if kind == 'char':
        return _atom('c', _char_value(form.value))
    if kind in ('list', 'vector'):
        return _collection('L' if kind == 'list' else 'V',
                           [canonical_form(x) for x in form.children])
    if kind == 'set':
        return _collection('S', sorted(canonical_form(x) for x in form.children))
    if kind == 'map':
        if len(form.children) % 2:
            raise UnsupportedShape('odd number of map entries')
        pairs = []
        for i in range(0,len(form.children),2):
            pairs.append(_collection('P', [canonical_form(form.children[i]),
                                          canonical_form(form.children[i+1])]))
        return _collection('M', sorted(pairs))
    raise UnsupportedShape('unsupported reader form ' + kind + ':' + form.value[:32])


def canonical_forms(forms) -> str:
    return _collection('T', [canonical_form(form) for form in forms])


def _literal_equality_key(form) -> str:
    """Clojure data equality within the supported, exact literal subset.

    Unlike the differential shape format, Clojure sequential equality ignores
    whether a sequence was written as a list or a vector. Apply this recursively
    inside map keys/values and set members, preserving scalar type distinctions.
    Unsupported leaves must invalidate the entire key rather than certify a
    partial comparison.
    """
    if form.kind in ('list', 'vector'):
        return _collection('Q', [_literal_equality_key(x) for x in form.children])
    if form.kind == 'set':
        return _collection('S', sorted(_literal_equality_key(x) for x in form.children))
    if form.kind == 'map':
        if len(form.children) % 2:
            raise UnsupportedShape('odd number of map entries')
        return _collection('M', sorted(
            _collection('P', [_literal_equality_key(form.children[i]),
                              _literal_equality_key(form.children[i+1])])
            for i in range(0, len(form.children), 2)))
    return canonical_form(form)


def literal_identity(form) -> str | None:
    """Safe equality key for duplicate literal map keys / set entries.

    Only a subset can be proved; returning None means unknown, not unique.
    """
    try:
        return _literal_equality_key(form)
    except (UnsupportedShape, ValueError, TypeError):
        return None
