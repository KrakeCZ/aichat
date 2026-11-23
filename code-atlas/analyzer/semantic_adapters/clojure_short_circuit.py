"""Source-only identity guards for Clojure's core ``and``/``or`` macros.

These macros short-circuit; treating each operand as an unconditional read or
straight-line CFG expression is inaccurate. Identity matters more than a name:
project Vars, locals, :refer :all, and :refer-clojure exclusions must never be
silently reinterpreted as core macros. No namespace is loaded and no macro is
expanded by this module.
"""
from __future__ import annotations


_NAMES = frozenset({'and', 'or'})


def candidates(forms, *, platform, referred, definition, top, sym):
    """Return source-supported spelling -> core identity, without certainty.

    Fully qualified names are recognized only if no require alias hides the
    literal core namespace. Unqualified names require an unambiguous literal
    ``ns`` clause without :refer :all or a colliding project declaration.
    Ambiguity fails closed, preserving the ordinary opaque flow fallback.
    """
    if platform not in ('clj', 'cljs'):
        return {}
    core = 'cljs.core' if platform == 'cljs' else 'clojure.core'
    ns_forms = [f for f in forms if top(f, 'ns')]
    if len(ns_forms) != 1:
        return {}
    ns = ns_forms[0]
    clauses = list(ns.children[2:])
    result = {core + '/' + name: name for name in _NAMES}
    # Any non-literal/ambiguous ns clause is an unsafe context for additional
    # name resolution. An explicit qualified core name remains recognized
    # unless a require-alias could hijack the namespace spelling.
    for clause in clauses:
        if not top(clause, ':require'):
            continue
        for lib in clause.children[1:]:
            if lib.kind != 'vector' or not lib.children:
                continue
            for i, part in enumerate(lib.children[1:], 1):
                if part.kind != 'atom' or part.value not in (':as', ':as-alias'):
                    continue
                if i + 1 < len(lib.children):
                    alias = lib.children[i + 1]
                    if alias.kind == 'atom' and alias.value == core and sym(lib.children[0]) != core:
                        result.clear()
                        return result
    # Project macros can bind the very same spelling at compile time. A
    # `:refer :all` may import names that source-only indexing cannot prove
    # absent, so it blocks all unqualified additions.
    unsafe_unqualified = False
    for clause in clauses:
        if top(clause, ':require'):
            for lib in clause.children[1:]:
                if lib.kind not in ('vector', 'list'):
                    continue
                values = lib.children
                if any(item.kind == 'atom' and item.value == ':all' for item in values):
                    unsafe_unqualified = True
        if top(clause, ':use'):
            unsafe_unqualified = True
    declared = {identified[0] for f in forms if (identified := definition(f))}
    referred_names = set(referred)
    enabled = set(_NAMES)
    options = [c for c in clauses if top(c, ':refer-clojure')]
    if len(options) > 1:
        unsafe_unqualified = True
    elif options:
        children = options[0].children[1:]
        if len(children) % 2:
            unsafe_unqualified = True
        else:
            keys = set()
            for idx in range(0, len(children), 2):
                key, value = children[idx:idx + 2]
                if key.kind != 'atom' or key.value in keys:
                    unsafe_unqualified = True
                    break
                keys.add(key.value)
                if key.value in (':exclude', ':only'):
                    if value.kind not in ('vector', 'list') or any(x.kind != 'atom' for x in value.children):
                        unsafe_unqualified = True
                        break
                    choices = {x.value for x in value.children}
                    if key.value == ':exclude':
                        enabled -= choices
                    else:
                        enabled &= choices
                elif key.value == ':rename':
                    # No compiler verified rename target. Do not assume the
                    # original (or renamed) macro remains available.
                    if value.kind != 'map' or len(value.children) % 2:
                        unsafe_unqualified = True
                        break
                    enabled -= {x.value for x in value.children[::2] if x.kind == 'atom'}
                else:
                    unsafe_unqualified = True
                    break
    if unsafe_unqualified:
        return result
    for name in enabled - declared - referred_names:
        result[name] = name
    return result
