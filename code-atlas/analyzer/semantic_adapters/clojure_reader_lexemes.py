"""Clojure JVM 1.12 lexical token guards; no execution or dependency loading.

Only surface-form syntax is tested here. Contextual keywords, read-time tags
and compiler phase are graded separately by clojure_reader_grammar.py.
"""
from __future__ import annotations

import re

_DECIMAL_INT = re.compile(r'[+-]?(?:0|[1-9][0-9]*)(?:N)?\Z')
_HEX_INT = re.compile(r'[+-]?0[xX][0-9A-Fa-f]+N?\Z')
_OCT_INT = re.compile(r'[+-]?0[0-7]+N?\Z')
_RADIX_INT = re.compile(r'([+-]?)([0-9]{1,2})[rR]([0-9A-Za-z]+)(N)?\Z')
_RATIO = re.compile(r'[+-]?[0-9]+/([0-9]+)\Z')
_FLOAT = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?M?\Z')


def _numeric_looking(token: str) -> bool:
    if token[0] in '0123456789':
        return True
    if token[0] in '+-' and len(token) > 1:
        tail = token[1:]
        return tail[0] in '0123456789' or (len(tail) > 1 and tail[0] == '.' and tail[1] in '0123456789')
    return len(token) > 1 and token[0] == '.' and token[1] in '0123456789'


def validate_atom(token: str, *, platform: str = 'clj') -> str | None:
    """Return precise source reader error type, or None for supported spelling.

    Never turns an invalid numeric literal into a Var/symbol. This guard does
    not perform number conversion; long literals cannot consume unbounded RAM.
    """
    if platform not in ('clj', 'cljs', 'shared'):
        raise ValueError('unsupported_lexical_platform')
    if platform == 'shared':
        # .cljc preserves both dialects before each branch's reader checks.
        # Do not reject a token merely because one dialect would reject it.
        java = validate_atom(token,platform='clj')
        js = validate_atom(token,platform='cljs')
        return java if java is not None and js is not None else None
    if not token:
        return 'empty_atom'
    if token.startswith(':'):
        if (token in (':','::') or token.startswith(':::') or
                (platform == 'clj' and '::' in token[2:]) or
                (platform == 'cljs' and token.count('/') > 1) or
                (token.endswith('/') and token != ':/') or
                (token.startswith(':/') and token != ':/')):
            return 'invalid_keyword_token'
        return None
    if ((token.startswith('/') and token != '/') or
            (token.endswith('/') and token != '/') or
            token.endswith(':') or
            (platform == 'clj' and '::' in token) or
            (platform == 'cljs' and '//' in token)):
        return 'invalid_symbol_token'
    if not _numeric_looking(token):
        return None
    if (_DECIMAL_INT.fullmatch(token) or _HEX_INT.fullmatch(token) or
            _OCT_INT.fullmatch(token) or
            (_FLOAT.fullmatch(token) and any(ch in token for ch in '.eEM'))):
        return None
    ratio = _RATIO.fullmatch(token)
    if ratio:
        if not ratio.group(1).strip('0'):
            return 'zero_denominator_ratio'
        return None
    radix = _RADIX_INT.fullmatch(token)
    if radix:
        base = int(radix.group(2))
        if not 2 <= base <= 36:
            return 'invalid_numeric_radix'
        digits = '0123456789abcdefghijklmnopqrstuvwxyz'
        if all(c.lower() in digits[:base] for c in radix.group(3)):
            return None
        return 'invalid_numeric_digit_for_radix'
    return 'invalid_number_literal'
