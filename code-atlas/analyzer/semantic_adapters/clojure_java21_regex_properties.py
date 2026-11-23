"""Static Java SE 21 Pattern Unicode-property name evidence (never executes JVM).

Only the portable public-name subset is *proven* by this table. Different JDK
versions and vendor-specific extensions require a separately pinned profile: this is preferable to incorrectly declaring source valid
or invalid. Source input is never sent to a runtime regex compiler.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import re

_POSIX=frozenset('Lower Upper ASCII Alpha Digit Alnum Punct Graph Print Blank Cntrl XDigit Space'.split())
_GC=frozenset('L Lu Ll Lt Lm Lo M Mn Mc Me N Nd Nl No P Pc Pd Ps Pe Pi Pf Po S Sm Sc Sk So Z Zs Zl Zp C Cc Cf Cs Co Cn LC'.split())
_BINARY=frozenset('Alphabetic Ideographic Letter Lowercase Uppercase Titlecase Punctuation Control White_Space Digit Hex_Digit Join_Control Noncharacter_Code_Point Assigned Emoji Emoji_Presentation Emoji_Modifier Emoji_Modifier_Base Emoji_Component Extended_Pictographic'.split())
_JAVA=frozenset('javaLowerCase javaUpperCase javaWhitespace javaMirrored javaDigit javaLetter javaLetterOrDigit javaAlphabetic javaIdeographic javaISOControl javaSpaceChar javaTitleCase javaUnicodeIdentifierStart javaUnicodeIdentifierPart javaJavaIdentifierStart javaJavaIdentifierPart javaIdentifierIgnorable'.split())


def _normalized(name: str) -> str:
    return re.sub(r'[\s_\-]', '', name).upper()


@lru_cache(maxsize=1)
def _jdk21_names() -> tuple[frozenset[str],frozenset[str]]:
    data=json.loads(Path(__file__).with_name('clojure_java21_unicode_property_names.json').read_text(encoding='utf8'))
    aliases=json.loads(Path(__file__).with_name('clojure_java21_unicode_script_aliases.json').read_text(encoding='utf8'))
    if aliases.get('alias_count') != 164 or len(set(aliases['aliases'])) != 164:
        raise ValueError('invalid_static_java21_unicode_script_alias_registry')
    blocks=json.loads(Path(__file__).with_name('clojure_java21_unicode_block_aliases.json').read_text(encoding='utf8'))
    if blocks.get('alias_count') != 756 or len(set(blocks['aliases'])) != 756:
        raise ValueError('invalid_static_java21_unicode_block_alias_registry')
    # Java UnicodeScript.forName is case-insensitive but does NOT erase
    # separators; UnicodeBlock.forName has its own explicit space/underscore
    # aliases. Dropping hyphens here falsely accepted malformed regexes.
    return (frozenset(x.upper() for x in data['names']['UnicodeScript']) |
            frozenset(x.upper() for x in aliases['aliases']),
            frozenset(x.upper() for x in blocks['aliases']))


def static_property_status(name: str) -> bool | None:
    """True = supported pinned Java SE 21 name; False = known unsupported.

    The Java 21 UnicodeScript aliases are complete for this pinned reference.
    An explicitly different JDK/version must select its own profile.
    """
    if not name or len(name)>256 or any(x in name for x in '\n\r\x00{}'):
        return False
    if name in _POSIX or name in _GC or name in _JAVA:
        return True
    script,block=_jdk21_names()
    binary={x.upper() for x in _BINARY}
    if name.startswith('java'):
        return False
    parts=name.split('=',1)
    if len(parts)==2:
        key,value=parts
        key=key.lower()
        if not value:return False
        if key in ('gc','general_category'):
            return value in _GC
        if key in ('sc','script'):
            if value.upper() in script:return True
            return False  # Complete Java21 UnicodeScript.forName aliases are indexed above
        if key in ('blk','block'):
            return value.upper() in block
        return False
    if name.startswith('In'):
        return name[2:].upper() in block
    if name.startswith('Is'):
        value=name[2:]
        if value in _GC or value.upper() in script or value.upper() in binary:
            return True
        return False  # Complete Java21 UnicodeScript.forName aliases are indexed above
    # Unknown (unprefixed) POSIX/binary names are not Java \p{...} syntax.
    return False
