"""Bounded *definite error* checks for Clojure JVM/CLJS reader regex literals.

Java Pattern syntax is more expressive than this static subset. This module
never compiles or runs a pattern from an analyzed project and never attempts
to infer matches. It only rejects reader-level syntax defects whose invalidity
is independently documented/tested. Not a full java.util.regex parser.
"""
from __future__ import annotations
import re

from .clojure_java21_regex_properties import static_property_status

_HEX = frozenset('0123456789abcdefABCDEF')
_ALLOWED_ESCAPE_LETTERS = frozenset('dDsSwWbBAZGzRrQEpPkKcht nvfuxaUXVHe'.replace(' ', ''))
_FLAG_LETTERS = frozenset('idmsuxU')
_QUANTIFIER = re.compile(r'([0-9]+)(?:,([0-9]*))?\Z')


def definitely_invalid_regex(pattern: str) -> str | None:
    """Return an evidence code for a certain syntax error, else None.

    Success means no known error was detected, NOT proof that the regex is
    valid for every Java/JavaScript engine or that it is safe to execute.
    """
    parens: list[int] = []
    classes: list[int] = []
    previous_atom = False
    quoted = False
    # A Java quantifier modifier (? = reluctant, + = possessive) is permitted
    # exactly once after a quantifier. A second quantifier is not an atom.
    just_quantified = False
    quantifier_modified = False
    quantifier_kind = None
    standalone_flag_group = False
    seen_named_groups: set[str] = set()
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if quoted:
            if pattern.startswith('\\E',i): quoted=False;i+=2;previous_atom=True;continue
            i+=1;continue
        if ch=='\\':
            if i+1 >= n:
                return 'regex_dangling_escape'
            e=pattern[i+1]
            if e=='Q':quoted=True;i+=2;previous_atom=True;just_quantified=False;continue
            if e=='E':return 'regex_quote_terminator_without_open_quote'
            if e=='c':
                if i+2>=n:return 'regex_incomplete_control_escape'
                i+=3;previous_atom=True;just_quantified=False;continue
            if e=='N':
                if i+2>=n or pattern[i+2]!='{':
                    return 'regex_missing_named_character_open_brace'
                end=pattern.find('}',i+3)
                if end<0 or end==i+3:
                    return 'regex_missing_or_empty_character_name'
                name=pattern[i+3:end]
                from .clojure_java21_unicode_char_names import jdk21_character_name_valid
                if not jdk21_character_name_valid(name):
                    return 'regex_unknown_unicode_character_name'
                i=end+1;previous_atom=True;just_quantified=False;continue
            if e in 'pP':
                if i+2>=n:return 'regex_incomplete_unicode_property'
                if pattern[i+2]=='{':
                    end=pattern.find('}',i+3)
                    if end<0 or end==i+3:return 'regex_empty_or_unclosed_unicode_property'
                    prop=pattern[i+3:end]
                    # A nonempty Unicode property name is not necessarily
                    # valid in the target JDK. These known invalid test
                    # properties are rejected; unfamiliar properties are
                    # graded as uncertain by the higher-level reader gate.
                    if static_property_status(prop) is False:
                        return 'regex_unrecognized_unicode_property'
                    i=end+1;previous_atom=True;just_quantified=False;continue
                if e=='p' or e=='P':
                    # The compact one-letter property form accepts only Java
                    # general-category names; arbitrary following letters
                    # aren't automatically legal property names.
                    if pattern[i+2] not in 'LMNPSZC':
                        return 'regex_unrecognized_unicode_property'
                i+=3;previous_atom=True;just_quantified=False;continue
            if e=='b' and pattern.startswith('{g}',i+2):
                i+=5;previous_atom=True;just_quantified=False;continue
            # Other `\b{n}` is parsed by java.util.regex.Pattern as a
            # boundary followed by a normal counted quantifier, NOT a bad
            # boundary mode. Invalid `{word}` is still rejected below.
            if e=='u':
                if i+6>n or any(x not in _HEX for x in pattern[i+2:i+6]):
                    return 'regex_invalid_unicode_escape'
                i+=6;previous_atom=True;continue
            if e=='x':
                if i+2 < n and pattern[i+2]=='{':
                    end=pattern.find('}',i+3)
                    if end<0 or end==i+3 or any(x not in _HEX for x in pattern[i+3:end]):
                        return 'regex_invalid_hex_escape'
                    codepoint=pattern[i+3:end].lstrip('0') or '0'
                    # Python 3.11 rejects huge int literals by default. A
                    # project-supplied regex must never crash the Atlas reader.
                    if len(codepoint)>6 or (len(codepoint)==6 and codepoint.upper()>'10FFFF'):
                        return 'regex_codepoint_out_of_range'
                    i=end+1;previous_atom=True;just_quantified=False;continue
                if i+4>n or any(x not in _HEX for x in pattern[i+2:i+4]):
                    return 'regex_invalid_hex_escape'
                i+=4;previous_atom=True;continue
            if e=='0' and (i+2==n or pattern[i+2] not in '01234567'):
                return 'regex_invalid_octal_escape'
            if e=='k':
                # Named backreferences require a previous (?<name> group.
                if i+2 >= n or pattern[i+2]!='<':
                    return 'regex_invalid_named_backreference'
                end=pattern.find('>',i+3)
                if end<0 or ('(?<'+pattern[i+3:end]+'>') not in pattern[:i]:
                    return 'regex_unknown_named_backreference'
                i=end+1;previous_atom=True;continue
            if e.isascii() and e.isalpha() and e not in _ALLOWED_ESCAPE_LETTERS:
                return 'regex_unknown_escape'
            i+=2;previous_atom=True;just_quantified=False;continue
        if ch=='[':
            classes.append(i)
            i+=1;previous_atom=False;just_quantified=False;continue
        if ch==']' and classes:
            start=classes.pop()
            if i==start+1 or (i==start+2 and pattern[start+1]=='^'):
                return 'regex_empty_character_class'
            i+=1;previous_atom=True;just_quantified=False;continue
        if classes:
            if ch=='-' and i+1<n and i>classes[-1]+1 and pattern[i+1]!=']':
                # Class ranges only compare single codepoints. Escaped
                # predefined *sets* on the left make '-' a literal (\d-a),
                # but a set on the right of a literal (`a-\d`) is invalid.
                prev=pattern[i-1]
                escaped_left=(i>=2 and pattern[i-2]=='\\')
                left_set=(escaped_left and prev in 'dDsSwW') or prev in '}]'
                # A `\p{...}` property escape ends with `}`; it denotes a
                # predefined class, not a character to be ranged.
                if prev == '}' and pattern.rfind('\\p{', classes[-1]+1, i)>=0:
                    left_set=True
                # In `[a-z--0-9]` the next hyphen follows a completed range;
                # Java treats it as a literal separator, not a z-to-`-` range.
                range_ended=(i>=classes[-1]+4 and pattern[i-2]=='-' and
                             pattern[i-3] not in ('[','\\'))
                leading_negation=(prev=='^' and i==classes[-1]+2)
                left_single=not (left_set or range_ended or leading_negation)
                right=pattern[i+1]
                if left_single and right=='\\' and i+2<n and pattern[i+2] in 'dDsSwWpP':
                    return 'regex_range_ends_with_predefined_class'
                if left_single and right not in ('\\','[') and ord(prev)>ord(right):
                    # Literal reversed ranges include Unicode/non-ASCII and
                    # punctuation; don't limit the check to ASCII letters.
                    return 'regex_reversed_character_range'
            i+=1;continue
        if ch=='(':
            if pattern.startswith('(?#',i):return 'regex_unsupported_group_comment'
            if pattern.startswith('(?|',i) or pattern.startswith('(?(',i):
                return 'regex_invalid_group_prefix'
            if pattern.startswith('(?',i) and i+2<n and pattern[i+2] in _FLAG_LETTERS.union({'-'}):
                end=min((v for v in (pattern.find(')',i+2),pattern.find(':',i+2)) if v>=0),default=-1)
                if end>=0:
                    flags=pattern[i+2:end]
                    if flags.count('-')>1 or any(c not in (_FLAG_LETTERS | {'-'}) for c in flags):
                        return 'regex_invalid_inline_flag'
            if pattern.startswith('(?',i) and i+2<n:
                modifier=pattern[i+2]
                if modifier!=')' and modifier not in '=!<:>idmsuxU-':
                    return 'regex_invalid_group_prefix'
            if pattern.startswith('(?<',i) and not pattern.startswith(('(?<=','(?<!'),i):
                end=pattern.find('>',i+3)
                if end<=i+3 or not re.fullmatch(r'[A-Za-z][A-Za-z0-9]*',pattern[i+3:end]):
                    return 'regex_invalid_named_group'
                if end>i+3:
                    name=pattern[i+3:end]
                    if name in seen_named_groups:
                        return 'regex_duplicate_named_group'
                    seen_named_groups.add(name)
            if pattern.startswith('(?',i):
                # Java flags may be enabled and/or disabled exactly once.
                # Do not parse lookarounds or named/atomic groups as flags.
                end=min([v for v in (pattern.find(':',i+2),pattern.find(')',i+2)) if v>=0],default=-1)
                if end>i+2 and pattern[i+2] in _FLAG_LETTERS and '--' in pattern[i+2:end]:
                    return 'regex_duplicate_flag_disable_separator'
            if pattern.startswith('(?',i) and i+2<n and pattern[i+2] in _FLAG_LETTERS.union({'-'}):
                end=min((v for v in (pattern.find(')',i+2),pattern.find(':',i+2)) if v>=0),default=-1)
                if end>=0:
                    flags=pattern[i+2:end]
                    if flags.count('-')>1 or any(c not in (_FLAG_LETTERS | {'-'}) for c in flags):
                        return 'regex_invalid_inline_flag'
            if pattern.startswith('(?',i) and i+2<n:
                modifier=pattern[i+2]
                if modifier.isascii() and modifier.isalpha() and modifier not in _FLAG_LETTERS:
                    return 'regex_invalid_inline_flag'
            parens.append(i)
            previous_atom=False;just_quantified=False;standalone_flag_group=False;i+=1;continue
        if ch==')':
            if not parens:return 'regex_unmatched_closing_parenthesis'
            opening=parens.pop()
            # A standalone inline flag switch `(?i)` changes the Pattern
            # mode; it is not a regex atom and cannot be quantified. Scoped
            # flag groups `(?i:a)` ARE atoms and may be quantified.
            flag_only = (pattern[opening:i+1]=='(?)' or bool(re.fullmatch(
                r'\(\?[idmsuxU]+(?:-[idmsuxU]+)?\)',pattern[opening:i+1])))
            previous_atom=not flag_only
            standalone_flag_group=flag_only
            just_quantified=False;i+=1;continue
        if ch=='|':previous_atom=False;just_quantified=False;i+=1;continue
        if ch in ('*','+','?'):
            # A '?' immediately after '(' opens a group modifier, not a
            # quantifier (?:), (?=), (?<name>), (?i), ... .
            if ch=='?' and i>0 and pattern[i-1]=='(':
                i+=1;continue
            if just_quantified:
                if ch=='*' or quantifier_modified:
                    return 'regex_nested_quantifier'
                quantifier_modified=True;i+=1;continue
            if not previous_atom:return 'regex_quantifier_without_atom'
            just_quantified=True;quantifier_modified=False;quantifier_kind=ch
            standalone_flag_group=False
            i+=1;previous_atom=True;continue
        if ch=='{':
            if just_quantified and quantifier_kind=='{':
                return 'regex_nested_quantifier'
            end=pattern.find('}',i+1)
            if end<0:
                return 'regex_incomplete_quantifier'
            interior=pattern[i+1:end]
            q=_QUANTIFIER.fullmatch(interior)
            if q:
                if not previous_atom and not standalone_flag_group:return 'regex_quantifier_without_atom'
                lo=q.group(1).lstrip('0') or '0'
                hi=(q.group(2).lstrip('0') or '0') if q.group(2) not in (None,'') else None
                # Lexical integer bounds avoid converting attacker-controlled
                # thousands-digit numbers, with bounded RAM/CPU use.
                def exceeds_i32(value):
                    return (len(value)>10 or (len(value)==10 and value>'2147483647'))
                if exceeds_i32(lo) or (hi is not None and exceeds_i32(hi)):
                    return 'regex_quantifier_exceeds_java_integer'
                if hi is not None and (len(hi)<len(lo) or (len(hi)==len(lo) and hi<lo)):
                    return 'regex_reversed_quantifier_bounds'
                just_quantified=True;quantifier_modified=False;quantifier_kind='{'
                standalone_flag_group=False
                previous_atom=True;i=end+1;continue
            return 'regex_invalid_brace_syntax'
        if ch=='^' or ch=='$':
            # Java Pattern permits quantifying anchors ($+, ^*).
            previous_atom=True
        else:
            previous_atom=True
        just_quantified=False;quantifier_modified=False;standalone_flag_group=False
        i+=1
    if parens:return 'regex_unclosed_parenthesis'
    if classes:return 'regex_unclosed_character_class'
    return None


def regex_has_version_dependent_property(pattern: str) -> bool:
    """True for explicit Unicode aliases not proven by the pinned JDK21 table.

    Quoted and double-escaped literals are ignored. A source-side uncertain
    alias must never be promoted to official accepted grammar by guessing.
    """
    i = 0
    quoted = False
    while i < len(pattern):
        if pattern.startswith('\\E',i) and quoted:
            quoted = False
            i += 2
            continue
        if quoted:
            i += 1
            continue
        if pattern.startswith('\\Q',i):
            quoted = True
            i += 2
            continue
        if pattern[i] == '\\' and i+1<len(pattern):
            if pattern[i+1] in 'pP' and pattern.startswith('{',i+2):
                j=pattern.find('}',i+3)
                if j>=0:
                    prop=pattern[i+3:j]
                    if static_property_status(prop) is None:
                        return True
                    i=j+1
                    continue
            i += 2
            continue
        i += 1
    return False
