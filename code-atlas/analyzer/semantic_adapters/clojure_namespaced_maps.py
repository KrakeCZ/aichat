"""Source-only Clojure namespaced-map syntax and key transformation.

No namespace is invented: auto-resolved #:: prefixes require an explicitly
supplied source namespace/alias. Only direct keys are rewritten, never values
or keys in nested maps. All returned forms retain physical source ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .clojure_shape import literal_identity


class NamespaceMapUnavailable(ValueError):
    pass


# A conservative namespace/alias symbol token subset: reject numeric,
# qualified, reader-like or malformed identifiers rather than infer a namespace.
_NAME = re.compile(r'[^\W\d][\w.?!*+\-]*\Z', re.UNICODE)
# Reader symbols/keyword names can be punctuation-only (+, -, *, ?, .) and
# contain a trailing #. The lexer has already checked token boundaries.
_NON_NAME = re.compile(r'[+\-]?(?:\d|\.\d)')


def _key_name(value: str) -> bool:
    return bool(value and not _NON_NAME.match(value) and
                not any(c in value for c in ' \t\r\n,;()[]{}\"\\'))



def prefix_valid(prefix: str) -> bool:
    if not prefix.startswith('#:'):
        return False
    auto = prefix.startswith('#::')
    name = prefix[3:] if auto else prefix[2:]
    return (auto and not name) or bool(_NAME.fullmatch(name))


def _rewrite_key(key, namespace: str):
    """Rewrite one unqualified keyword/symbol key, preserving source offsets."""
    from .clojure_forms import Form

    if key.kind != 'atom':
        return key
    token = key.value
    if token.startswith('::'):
        # Alias-resolved keywords need a separate namespace resolver. Do not
        # assert a lexical value for them in a namespaced map.
        raise NamespaceMapUnavailable('auto_resolved_key_requires_alias_resolution')
    keyword = token.startswith(':')
    spelling = token[1:] if keyword else token
    if not spelling or spelling in ('nil', 'true', 'false', '/'):
        return key
    if '/' in spelling:
        old_ns, sep, name = spelling.partition('/')
        if not _key_name(old_ns) or not _key_name(name):
            return key  # numbers/ratios and malformed reader tokens are not names
        new_spelling = name if old_ns == '_' else spelling
    else:
        if not _key_name(spelling):
            return key  # numeric literals and unknown tokens are not symbols
        new_spelling = namespace + '/' + spelling
    result = ':' + new_spelling if keyword else new_spelling
    return Form(key.kind, result, key.children, key.line, key.col,
                key.metadata, key.end_line, key.end_col)


def resolve_namespace(prefix: str, *, current_namespace: str | None = None,
                      aliases: dict[str, str] | None = None) -> str | None:
    if not prefix_valid(prefix):
        raise NamespaceMapUnavailable('invalid_namespaced_map_prefix')
    if not prefix.startswith('#::'):
        return prefix[2:]
    alias = prefix[3:]
    if not alias:
        return current_namespace
    target = (aliases or {}).get(alias)
    if target is None:
        raise NamespaceMapUnavailable('unknown_namespaced_map_alias')
    return target


def normalize_map(form, *, current_namespace: str | None = None,
                  aliases: dict[str, str] | None = None):
    """Return resolved map Form or None where namespace evidence is missing.

    Raise for a certain duplicate key; never compare opaque/unknown values.
    """
    from .clojure_forms import Form

    if form.kind != 'namespaced_map':
        raise NamespaceMapUnavailable('not_namespaced_map')
    namespace = resolve_namespace(form.value, current_namespace=current_namespace,
                                  aliases=aliases)
    if namespace is None:
        return None
    if len(form.children) % 2:
        raise NamespaceMapUnavailable('odd_namespaced_map_entries')
    children = list(form.children)
    seen: set[str] = set()
    for i in range(0, len(children), 2):
        children[i] = _rewrite_key(children[i], namespace)
        key = literal_identity(children[i])
        if key is not None:
            if key in seen:
                raise NamespaceMapUnavailable('duplicate_namespaced_map_key')
            seen.add(key)
    return Form('map',form.value,tuple(children),form.line,form.col,
                form.metadata,form.end_line,form.end_col)
