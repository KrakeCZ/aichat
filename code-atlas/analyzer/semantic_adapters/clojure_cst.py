"""Source-anchored, lossless-*coverage* CST token index for Clojure source.

This is NOT an official Clojure reader CST, compiler AST or macro-expanded tree.
The existing bounded structural reader must accept the entire source first.
Every byte/codepoint of its decoded input is accounted for by token and trivia
leaves. The original source is needed to reconstruct contents; exported nodes
contain hashes and ranges, not the user's source text. Uncertain reader constructs
remain opaque and are never considered resolved by this index.
"""
from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256

MAX_CST_TOKENS = 5000
MAX_CST_NODES = 16000
MAX_CST_DEPTH = 80
MAX_CST_OUTPUT_BYTES = 2_000_000


class CSTUnavailable(ValueError):
    pass


_OPEN_CLOSE = {'(': ')', '[': ']', '{': '}', '#{': '}', '#(': ')'}
_OPEN_KIND = {'(': 'list', '[': 'vector', '{': 'map', '#{': 'set', '#(': 'reader_function'}


def _trivia_parts(text: str, begin: int, end: int):
    """Split only lexer-elided trivia; fail closed on unaccounted content."""
    i = begin
    while i < end:
        start = i
        if text[i] == ';' or text.startswith('#!', i):
            kind = 'comment' if text[i] == ';' else 'shebang_comment'
            nl = text.find('\n', i, end)
            i = end if nl == -1 else nl
            # Next newline is separately represented as whitespace.
        elif text[i].isspace() or text[i] == ',':
            kind = 'separator'
            while i < end and (text[i].isspace() or text[i] == ','):
                i += 1
        else:
            raise CSTUnavailable('unaccounted_lexer_gap')
        if i <= start:
            raise CSTUnavailable('zero_length_trivia')
        yield (kind, start, i)


def _line_starts(text: str):
    starts = [0]
    for i, char in enumerate(text):
        if char == '\n':
            starts.append(i + 1)
    return starts


def _utf8_offsets(text: str):
    offsets = [0] * (len(text) + 1)
    size = 0
    for i, char in enumerate(text):
        size += len(char.encode('utf-8'))
        offsets[i + 1] = size
    return offsets


def build_source_cst(text: str, tokens, *, source_sha256: str, platform: str,
                     official_reader_evidence=None):
    """Return a complete bounded source-coordinate CST index or unavailable.

    The input `tokens` MUST be from clojure_forms.lex for exactly `text`, and
    `_Parser(tokens).all()` MUST already have succeeded. No user code runs.
    The returned structure has a single root, form nodes, token leaves, and
    comment/separator leaves. Ranges are 0-based Unicode/UTF-8 end-exclusive;
    line/column are 1-based Unicode end-exclusive (consistent with the lexer).
    """
    if platform not in ('clj', 'cljs', 'shared'):
        return {'status': 'unavailable', 'reason': 'unsupported_platform'}
    if len(tokens) > MAX_CST_TOKENS:
        return {'status': 'unavailable', 'reason': 'cst_token_limit',
                'token_count': len(tokens), 'limit': MAX_CST_TOKENS}
    try:
        return _construct(text, tokens, source_sha256, platform, official_reader_evidence)
    except (CSTUnavailable, UnicodeEncodeError, RecursionError) as exc:
        return {'status': 'unavailable', 'reason': (
            str(exc) if isinstance(exc, CSTUnavailable) else 'cst_encoding_or_depth_invalid')[:96]}


def _construct(text, tokens, source_sha256, platform, official_reader_evidence):
    starts = _line_starts(text)
    utf8 = _utf8_offsets(text)
    # Browser/LSP coordinates count UTF-16 code units, Python spans count
    # Unicode code points, source hashes count original bytes. Do not conflate.
    utf16 = [0]
    for ch in text:
        utf16.append(utf16[-1] + (2 if ord(ch) > 0xFFFF else 1))
    n = len(text)
    significant = []
    previous = 0
    for token in tokens:
        if not all(isinstance(v, int) for v in (token.line, token.col, token.end_line, token.end_col)):
            raise CSTUnavailable('token_coordinate_missing')
        if not (1 <= token.line <= len(starts) and 1 <= token.end_line <= len(starts)):
            raise CSTUnavailable('token_line_invalid')
        b = starts[token.line - 1] + token.col - 1
        e = starts[token.end_line - 1] + token.end_col - 1
        if not (previous <= b < e <= n):
            raise CSTUnavailable('token_interval_invalid')
        significant.append((b, e))
        previous = e

    nodes = []
    def location(offset):
        line = bisect_right(starts, offset)
        return (line, offset - starts[line - 1] + 1)

    def node(kind, start, end, parent, *, category=None):
        if len(nodes) >= MAX_CST_NODES:
            raise CSTUnavailable('cst_node_limit')
        if not (0 <= start <= end <= n):
            raise CSTUnavailable('node_range_invalid')
        ident = len(nodes)
        line, col = location(start)
        end_line, end_col = location(end)
        entry = {'id': ident, 'parent': parent, 'kind': kind,
                 'range': [start, end], 'utf8_range': [utf8[start], utf8[end]],
                 'utf16_range': [utf16[start], utf16[end]],
                 'line': line, 'column': col, 'end_line': end_line, 'end_column': end_col}
        if category is not None:
            entry['category'] = category
        nodes.append(entry)
        return ident

    root = node('source', 0, n, None)
    owners = [None] * len(tokens)
    pos = 0
    def parse(parent: int, depth: int):
        nonlocal pos
        if depth > MAX_CST_DEPTH:
            raise CSTUnavailable('cst_depth_limit')
        if pos >= len(tokens):
            raise CSTUnavailable('missing_reader_operand')
        idx = pos
        tok = tokens[pos]
        b, e = significant[pos]
        pos += 1
        if tok.kind == 'close':
            raise CSTUnavailable('unmatched_close')
        if tok.kind in ('open', 'opaque_open'):
            ident = node('form', b, e, parent, category=_OPEN_KIND.get(tok.value, 'unknown_collection'))
            owners[idx] = ident
            expected = _OPEN_CLOSE.get(tok.value)
            if expected is None:
                raise CSTUnavailable('unknown_collection_delimiter')
            while pos < len(tokens) and tokens[pos].kind != 'close':
                parse(ident, depth + 1)
            if pos >= len(tokens) or tokens[pos].value != expected:
                raise CSTUnavailable('collection_delimiter_mismatch')
            owners[pos] = ident
            nodes[ident]['range'][1] = significant[pos][1]
            # Subtree parent range must include the entire closing delimiter.
            stop = significant[pos][1]
            nodes[ident]['utf8_range'][1] = utf8[stop]
            nodes[ident]['utf16_range'][1] = utf16[stop]
            nodes[ident]['end_line'], nodes[ident]['end_column'] = location(stop)
            pos += 1
            return ident
        if tok.kind in ('discard', 'meta', 'opaque_prefix', 'namespaced_prefix'):
            category = ('reader_discard' if tok.kind == 'discard' else
                        'metadata' if tok.kind == 'meta' else
                        'namespaced_map' if tok.kind == 'namespaced_prefix' else
                        'opaque_reader_prefix')
            ident = node('form', b, e, parent, category=category)
            owners[idx] = ident
            operand_count = 2 if tok.kind == 'meta' else 1
            last_child = None
            for _ in range(operand_count):
                last_child = parse(ident, depth + 1)
            stop = nodes[last_child]['range'][1]
            nodes[ident]['range'][1] = stop
            nodes[ident]['utf8_range'][1] = utf8[stop]
            nodes[ident]['utf16_range'][1] = utf16[stop]
            nodes[ident]['end_line'], nodes[ident]['end_column'] = location(stop)
            return ident
        ident = node('form', b, e, parent, category=tok.kind)
        owners[idx] = ident
        return ident

    while pos < len(tokens):
        parse(root, 1)
    if any(owner is None for owner in owners):
        raise CSTUnavailable('unowned_token')

    # Append physical token and trivia leaves. They are source-addressable and
    # SHA-256-verified, but do not embed user project source fragments.
    for idx, (b, e) in enumerate(significant):
        tok = tokens[idx]
        owner = owners[idx]
        leaf = node('token', b, e, owner, category=tok.kind)
        nodes[leaf]['text_sha256'] = sha256(text[b:e].encode('utf-8')).hexdigest()
    def ancestor_chain(ident):
        chain = []
        while ident is not None:
            chain.append(ident)
            ident = nodes[ident]['parent']
        return chain
    def lca(left, right):
        if left is None or right is None:
            return root
        right_set = set(ancestor_chain(right))
        return next(x for x in ancestor_chain(left) if x in right_set)

    previous = 0
    leaves = []
    for idx, (b, e) in enumerate(significant):
        if previous < b:
            parent = lca(owners[idx - 1] if idx else None, owners[idx])
            for kind, start, end in _trivia_parts(text, previous, b):
                leaves.append((kind, start, end, parent))
        previous = e
    if previous < n:
        for kind, start, end in _trivia_parts(text, previous, n):
            leaves.append((kind, start, end, root))
    if not tokens and text:
        leaves = [(kind, b, e, root) for kind,b,e in _trivia_parts(text, 0, n)]

    for kind,b,e,parent in leaves:
        leaf = node('trivia', b, e, parent, category=kind)
        nodes[leaf]['text_sha256'] = sha256(text[b:e].encode('utf-8')).hexdigest()

    segments = sorted((v['range'][0], v['range'][1]) for v in nodes
                      if v['kind'] in ('token','trivia'))
    cursor = 0
    for b, e in segments:
        if cursor != b or e <= b:
            raise CSTUnavailable('cst_physical_coverage_gap_or_overlap')
        cursor = e
    if cursor != n:
        raise CSTUnavailable('cst_physical_coverage_incomplete')
    for entry in nodes[1:]:
        parent = nodes[entry['parent']]
        if not (parent['range'][0] <= entry['range'][0] <= entry['range'][1] <= parent['range'][1]):
            raise CSTUnavailable('cst_invalid_parent_span')

    # Avoid a potentially unbounded large file-size multiplier in exports.
    import json
    result = {
        'status': 'complete_source_coverage_structural_subset',
        'schema_version': 1,
        'source_sha256': source_sha256,
        'decoded_source_sha256': sha256(text.encode('utf-8')).hexdigest(),
        'platform': platform,
        'coordinates': 'unicode_codepoints_zero_based_and_utf8_bytes_end_exclusive',
        'reader_authority': 'clojure_structural_lexer_not_official_ast',
        'source_length': n,
        'decoded_utf8_length': utf8[-1],
        'decoded_utf16_codeunit_length': utf16[-1],
        'token_count': len(significant),
        'trivia_count': len(leaves),
        'form_count': sum(entry['kind'] == 'form' for entry in nodes),
        'node_count': len(nodes),
        'source_coverage': 'all_decoded_characters_exactly_once_by_token_or_trivia_leaves',
        'nodes': nodes,
    }
    if official_reader_evidence:
        # Official read success authenticates only the complete input stream,
        # not the particular structure of our independent source index.
        observations = list(official_reader_evidence.values())
        result['whole_source_officially_accepted'] = (
            bool(observations) and all(v.get('status') == 'accepted' for v in observations))
    if len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > MAX_CST_OUTPUT_BYTES:
        raise CSTUnavailable('cst_export_byte_limit')
    return result
