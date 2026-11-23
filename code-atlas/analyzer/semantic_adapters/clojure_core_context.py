"""Conservative namespace overrides for structural core-macro scope rules."""


def shadowed_core_names(forms, referred):
    from .clojure_forms import _definition, _top, _sym
    from .clojure_scope import CORE_SCOPE_MACROS
    shadowed=set(referred)
    shadowed.update(d[0] for f in forms if (d:=_definition(f)))
    declared_or_referred=set(shadowed)
    all_core={n for n in CORE_SCOPE_MACROS if '/' not in n}
    ns=[f for f in forms if _top(f,'ns')]
    if len(ns)!=1:
        return frozenset(shadowed|all_core)
    for clause in ns[0].children[2:]:
        if not _top(clause,':refer-clojure'):
            continue
        parts=clause.children[1:]
        if len(parts)%2:
            shadowed.update(all_core);continue
        for i in range(0,len(parts),2):
            option=_sym(parts[i]);value=parts[i+1]
            if option==':exclude' and value.kind in ('vector','list'):
                shadowed.update(_sym(x) for x in value.children)
            elif option==':only' and value.kind in ('vector','list'):
                shadowed.update(all_core-{_sym(x) for x in value.children})
            elif option==':rename' and value.kind=='map':
                shadowed.update(_sym(x) for x in value.children[::2])
            else:
                shadowed.update(all_core)
    for alias,identity in core_macro_aliases(forms).items():
        if identity is not None and '/' not in alias and alias not in declared_or_referred:
            shadowed.discard(alias)
    return frozenset(shadowed)


def core_macro_aliases(forms):
    """Literal core scope identities; None marks a known ambiguous spelling.

    In particular a filtered-out rename must not be used to infer bindings.
    No namespace is loaded and no runtime Var identity is claimed.
    """
    from .clojure_forms import _top, _sym
    from .clojure_scope import CORE_SCOPE_MACROS
    from .clojure_libspecs import simple_symbol
    aliases={}
    ns=[f for f in forms if _top(f,'ns')]
    clauses=[c for f in ns for c in f.children[2:] if _top(c,':refer-clojure')]
    for clause in clauses:
        parts=clause.children[1:];options={};valid=len(ns)==1 and len(clauses)==1 and len(parts)%2==0
        for i in range(0,len(parts)-1,2):
            key=_sym(parts[i]);value=parts[i+1]
            if key in options or key not in (':exclude',':only',':rename'):valid=False
            options[key]=value
            if key in (':exclude',':only'):
                if value.kind not in ('vector','list') or any(not simple_symbol(x) for x in value.children):valid=False
            elif key==':rename':
                if value.kind!='map' or any(not simple_symbol(x) for x in value.children):valid=False
        excluded={_sym(x) for x in options[':exclude'].children} if ':exclude' in options else set()
        included={_sym(x) for x in options[':only'].children} if ':only' in options else None
        destinations={}
        for i in range(0,len(parts)-1,2):
            if _sym(parts[i])==':rename' and parts[i+1].kind=='map':
                for alias in parts[i+1].children[1::2]:
                    destinations[alias.value]=destinations.get(alias.value,0)+1
        # Visit every rename occurrence, including repeated invalid options:
        # all affected aliases must stay opaque rather than the last one win.
        for i in range(0,len(parts)-1,2):
            if _sym(parts[i])!=':rename' or parts[i+1].kind!='map':continue
            entries=parts[i+1].children
            for j in range(0,len(entries)-1,2):
                original=_sym(entries[j]);alias=entries[j+1]
                if original not in CORE_SCOPE_MACROS or not simple_symbol(alias):continue
                usable=valid and original not in excluded and (included is None or original in included)
                usable=usable and destinations.get(alias.value)==1
                # A rename cannot overwrite another still-referred core Var.
                if alias.value in CORE_SCOPE_MACROS and alias.value!=original:
                    usable=usable and (alias.value in excluded or
                        (included is not None and alias.value not in included))
                identity='clojure.core/'+original if usable else None
                aliases[alias.value]=identity if alias.value not in aliases else None
    # Only validated, non-conflicting require aliases can name a core namespace.
    # No project namespace is guessed to be core based on its short spelling.
    from .clojure_forms import _namespace, _namespace_alias_guards
    for platform in ('clj', 'cljs'):
        _, required_aliases, _, issue=_namespace(forms, platform)
        guards=_namespace_alias_guards(forms, platform)
        if issue:continue
        for alias, namespace in required_aliases.items():
            if alias in guards or namespace not in ('clojure.core','cljs.core'):continue
            for name in CORE_SCOPE_MACROS:
                if '/' not in name and namespace+'/'+name in CORE_SCOPE_MACROS:
                    aliases[alias+'/'+name]=namespace+'/'+name
    return aliases
