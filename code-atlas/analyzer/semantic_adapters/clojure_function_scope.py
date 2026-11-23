"""Literal fn contracts only: no eval, expansion, branch or assertion guarantee."""
from __future__ import annotations


def function_contract_plan(params, expressions, *, expand_contracts=True):
    """Return (pre, body, post), or None for an unsupported contract shape.

    JVM fn uses a leading map as conditions only when a body follows it.
    Otherwise parameter-vector metadata supplies the conditions. Raw fn* has
    no such macro semantics. Unknown map entries are data, not expressions.
    Multiple overlapping metadata contracts remain opaque instead of guessing
    the reader's merge precedence. This is a candidate plan, not macroexpansion.
    """
    if not expand_contracts:
        return (), expressions, ()
    conditions={}
    if len(expressions)>1 and expressions[0].kind=='map':
        maps=(expressions[0],)
        expressions=expressions[1:]
    else:
        maps=[]
        for metadata in params.metadata:
            if metadata.kind=='map':maps.append(metadata)
            elif metadata.kind not in ('atom','string'):
                return None
    for mapping in maps:
        for key,value in zip(mapping.children[::2],mapping.children[1::2]):
            if key.kind=='atom' and key.value in (':pre',':post'):
                if key.value in conditions:return None
                conditions[key.value]=value
    parts=[]
    for key in (':pre',':post'):
        value=conditions.get(key)
        if value is None or (value.kind=='atom' and value.value in ('nil','false')):
            parts.append(())
        elif value.kind in ('vector','list'):
            parts.append(value.children)
        else:
            return None
    return parts[0],expressions,parts[1]
