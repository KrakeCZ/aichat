"""Inert byte-faithful CST recovery for rejected Clojure/CLJS source.

This does not replace the structural reader and does NOT accept malformed code
as Clojure. It preserves source coverage and coordinates for Inside File even if
UTF-8 decoding, lexical validation or form parsing fails. It never evaluates
project macros, reader tags, data, syntax quote or names; NO semantic graph
edges or declaration nodes may be derived from recovery tokens.

The source text itself is not embedded in the output: physical leaves contain
only byte/codepoint offsets and hashes. A file above the independent limits is
explicitly unavailable, never partially presented as complete.
"""
from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256
import json

MAX_RECOVERY_BYTES = 750_000
MAX_RECOVERY_LEAVES = 16000
MAX_RECOVERY_JSON_BYTES = 2_000_000


class RecoveryUnavailable(ValueError):
    """A bounded lossless recovery index cannot safely be constructed."""


def _segments(text: str):
    """Inert lexical segmentation, not a language parser or accepted tokenization.

    The scanner only recognizes physical separators, quotes, comments,
    delimiters and opaque fragments. An invalid string/escape remains one
    physical fragment. It cannot create declaration/call/namespace evidence.
    """
    index = 0
    limit = len(text)
    while index < limit:
        begin = index
        char = text[index]
        if char.isspace() or char == ',':
            index += 1
            while index < limit and (text[index].isspace() or text[index] == ','):
                index += 1
            kind = 'separator'
        elif char == ';' or text.startswith('#!', index):
            index = text.find('\n', index)
            if index < 0:
                index = limit
            kind = 'comment' if char == ';' else 'shebang_comment'
        elif char == '"' or text.startswith('#"', index):
            is_regex = char == '#'
            index += 2 if is_regex else 1
            closed = False
            while index < limit:
                if text[index] == '\\':
                    index += 2 if index + 1 < limit else 1
                elif text[index] == '"':
                    index += 1
                    closed = True
                    break
                else:
                    index += 1
            kind = ('regex_literal' if is_regex else 'string_literal') if closed else 'unterminated_literal'
        elif char in '()[]{}':
            index += 1
            kind = 'collection_delimiter'
        elif text.startswith('#?@', index) or text.startswith('~@', index):
            index += 3 if text.startswith('#?@', index) else 2
            kind = 'reader_prefix'
        elif text.startswith(('#?', '#_', "#'", '#=', '#{', '#('), index):
            index += 2
            kind = ('blocked_read_eval' if text.startswith('#=', begin) else
                    'reader_collection_open' if text.startswith(('#{', '#('), begin) else
                    'reader_prefix')
        elif char in "'`~@^":
            index += 1
            kind = 'reader_prefix'
        elif char == '\\':
            index += 1
            if index < limit:
                index += 1
                if text[index - 1].isalnum():
                    while index < limit and text[index] not in ' \t\n\r,;()[]{}':
                        index += 1
            kind = 'character_literal_opaque'
        else:
            # No binding, type, symbol or call interpretation is possible here.
            index += 1
            while index < limit and not (
                text[index].isspace() or text[index] in ',;()[]{}"\'`~@^'
                or text.startswith('#!', index) or text.startswith('#"', index)
            ):
                index += 1
            kind = 'opaque_source_fragment'
        if index <= begin:
            raise RecoveryUnavailable('scanner_failed_to_advance')
        yield kind, begin, index


def recover_rejected_source_cst(raw: bytes, *, platform: str,
                                reader_error_category: str = 'structural_reader_rejected') -> dict:
    """Recover raw bytes + physical coordinates without evaluating source.

    Result `status=recovered_invalid_source_only` means lossless *physical*
    coverage, never valid grammar or source binding. For invalid UTF-8 the
    surrogateescape decoder is used only to retain a byte-to-offset map, not
    to interpret characters as valid Clojure text.
    """
    source_hash = sha256(raw).hexdigest()
    if platform not in ('clj', 'cljs', 'shared'):
        return {'status':'unavailable','reason':'unsupported_platform','source_sha256':source_hash}
    if len(raw) > MAX_RECOVERY_BYTES:
        return {'status':'unavailable','reason':'recovery_byte_limit','source_sha256':source_hash,
                'limit':MAX_RECOVERY_BYTES}
    try:
        return _recover(raw, platform, reader_error_category, source_hash)
    except (RecoveryUnavailable, UnicodeError, RecursionError) as exc:
        return {'status':'unavailable','reason':(
            str(exc) if isinstance(exc, RecoveryUnavailable) else 'recovery_encoding_invalid')[:96],
            'source_sha256':source_hash,'platform':platform}


def _recover(raw: bytes, platform: str, reader_error_category: str, source_hash: str) -> dict:
    bom = raw.startswith(b'\xef\xbb\xbf')
    initial = 3 if bom else 0
    text = raw[initial:].decode('utf-8', errors='surrogateescape')
    utf8 = [initial]
    utf16 = [0]
    line_starts = [0]
    has_invalid_utf8 = False
    for i, char in enumerate(text):
        if 0xdc80 <= ord(char) <= 0xdcff:
            has_invalid_utf8 = True
        utf8.append(utf8[-1] + len(char.encode('utf-8', errors='surrogateescape')))
        utf16.append(utf16[-1] + (2 if ord(char) > 0xffff else 1))
        if char == '\n':
            line_starts.append(i + 1)
    if utf8[-1] != len(raw):
        raise RecoveryUnavailable('source_byte_mapping_mismatch')

    def location(offset: int):
        line = bisect_right(line_starts, offset)
        return line, offset - line_starts[line - 1] + 1

    nodes = [{'id':0,'parent':None,'kind':'source','range':[0,len(text)],
              'utf8_range':[0,len(raw)],'utf16_range':[0,utf16[-1]],
              'line':1,'column':1,
              'end_line':location(len(text))[0],'end_column':location(len(text))[1]}]
    if bom:
        nodes.append({'id':len(nodes),'parent':0,'kind':'bom','category':'utf8_bom',
                      'range':[0,0],'utf8_range':[0,3],'utf16_range':[0,0],
                      'line':1,'column':1,'end_line':1,'end_column':1,
                      'text_sha256':sha256(raw[:3]).hexdigest()})
    for kind, begin, end in _segments(text):
        if len(nodes) >= MAX_RECOVERY_LEAVES + 2:
            raise RecoveryUnavailable('recovery_leaf_limit')
        lb,cb = location(begin)
        le,ce = location(end)
        b,e = utf8[begin],utf8[end]
        if b >= e:
            raise RecoveryUnavailable('recovery_leaf_empty')
        nodes.append({'id':len(nodes),'parent':0,
                      'kind':'trivia' if kind in ('separator','comment','shebang_comment') else 'token',
                      'category':kind,'range':[begin,end], 'utf8_range':[b,e],
                      'utf16_range':[utf16[begin],utf16[end]] if not has_invalid_utf8 else None,
                      'line':lb,'column':cb,'end_line':le,'end_column':ce,
                      'text_sha256':sha256(raw[b:e]).hexdigest()})
    physical = sorted((n['utf8_range'] for n in nodes[1:]), key=lambda v:v[0])
    offset = 0
    for b,e in physical:
        if b != offset or e <= b:
            raise RecoveryUnavailable('recovery_physical_overlap_or_gap')
        offset = e
    if offset != len(raw):
        raise RecoveryUnavailable('recovery_physical_incomplete')
    # These are physical delimiter observations, not compiler diagnostics or
    # a claim that other content is valid. Read-eval is never interpreted.
    diagnostics = []
    bracket_stack = []
    expected = {'(':')','[':']','{':'}','#{':'}','#(':')'}
    for leaf in nodes[1:]:
        category = leaf.get('category')
        if category not in ('collection_delimiter','reader_collection_open',
                            'unterminated_literal','blocked_read_eval'):
            continue
        b,e = leaf['utf8_range']
        spelling = raw[b:e].decode('ascii', errors='ignore') if e-b<=2 else ''
        issue = None
        if category == 'unterminated_literal':
            issue = 'unterminated_quoted_literal'
        elif category == 'blocked_read_eval':
            issue = 'read_eval_forbidden_not_executed'
        elif spelling in expected:
            bracket_stack.append((spelling,leaf))
        elif spelling in (')',']','}'):
            if not bracket_stack:
                issue = 'unmatched_closing_delimiter'
            elif expected[bracket_stack[-1][0]] == spelling:
                bracket_stack.pop()
            else:
                issue = 'mismatched_closing_delimiter'
                # Fail closed; retain the opening token for a separate missing
                # closer observation instead of inventing a valid collection.
        if issue and len(diagnostics)<64:
            diagnostics.append({'code':issue,'line':leaf['line'],
                                'column':leaf['column'],
                                'utf8_range':leaf['utf8_range'],
                                'source':'inert_recovery_scanner_not_compiler'})
    for _spelling, leaf in bracket_stack:
        if len(diagnostics)>=64:
            break
        diagnostics.append({'code':'opening_delimiter_without_matching_closer',
                            'line':leaf['line'],'column':leaf['column'],
                            'utf8_range':leaf['utf8_range'],
                            'source':'inert_recovery_scanner_not_compiler'})
    if has_invalid_utf8 and len(diagnostics)<64:
        for leaf in nodes[1:]:
            b,e=leaf['utf8_range']
            if any(0xdc80<=ord(c)<=0xdcff for c in text[leaf['range'][0]:leaf['range'][1]]):
                diagnostics.append({'code':'invalid_utf8_source_bytes',
                                    'line':leaf['line'],'column':leaf['column'],
                                    'utf8_range':leaf['utf8_range'],
                                    'source':'strict_utf8_decoding_failure'})
                break
    result = {
        'status':'recovered_invalid_source_only',
        'schema_version':1,
        'source_sha256':source_hash,
        'platform':platform,
        'reader_authority':'inert_recovery_lexer_not_valid_clojure_syntax',
        'reader_valid':False,
        'semantic_edges_allowed':False,
        'source_length':len(text),
        'raw_byte_length':len(raw),
        'decoded_utf16_codeunit_length':utf16[-1] if not has_invalid_utf8 else None,
        'raw_utf8_valid':not has_invalid_utf8,
        'bom_present':bom,
        'coordinates':'one_based_unicode_source_and_original_bytes_end_exclusive',
        'source_coverage':'all_original_bytes_once_in_token_trivia_or_bom_leaves',
        'reader_error_category':reader_error_category[:80],
        'diagnostics':diagnostics,
        'diagnostics_truncated':len(diagnostics)>=64,
        'node_count':len(nodes),
        'physical_leaf_count':len(nodes)-1,
        'nodes':nodes,
    }
    if len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode('utf-8')) > MAX_RECOVERY_JSON_BYTES:
        raise RecoveryUnavailable('recovery_export_byte_limit')
    return result
