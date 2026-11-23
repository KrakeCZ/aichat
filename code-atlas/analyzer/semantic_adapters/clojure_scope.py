"""Conservative lexical-scoping guard for *candidate* Clojure :refer calls.

Does not execute Clojure, macros or the project. Scans ordinary call-head
positions only when their lexical bindings can be accounted for. Unsupported
scope forms are skipped rather than treated as global references. The result
is not a full Clojure compiler, a semantic resolver or a CFG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .clojure_forms import Form


# These forms stay opaque to the context-free structural walker. The lexical
# walker below handles a documented subset BEFORE this gate. This distinction
# prevents a binding pattern from being mistaken for executable structure.
OPAQUE_BINDING_MACROS = frozenset({
    'for','doseq','dotimes','when-let','if-let','when-some','if-some',
    'when-first','with-open','binding','with-local-vars','as->',
    'with-redefs','with-redefs-fn','with-bindings','with-bindings*',
    'clojure.core/for','clojure.core/doseq','clojure.core/dotimes',
    'clojure.core/when-let','clojure.core/if-let',
    'clojure.core/when-some','clojure.core/if-some',
    'clojure.core/when-first','clojure.core/with-open',
    'clojure.core/binding','clojure.core/with-redefs',
    'clojure.core/as->',
})
_CORE_BINDINGS = frozenset({'if-let', 'if-some', 'when-let', 'when-some',
                            'when-first', 'dotimes', 'with-open', 'for',
                            'doseq', 'as->'})
CORE_BINDING_MACROS = {name: name for name in _CORE_BINDINGS}
CORE_BINDING_MACROS.update({'clojure.core/' + name: name for name in _CORE_BINDINGS})
# CLJS has no core with-open; keep that spelling opaque rather than infer JVM
# resource semantics. Imported CLJS conditional macros share these scopes.
CORE_BINDING_MACROS.update({'cljs.core/' + name: name
                           for name in _CORE_BINDINGS - {'with-open'}})
# All core macros whose argument evaluation this walker interprets. The same
# identity table must govern qualifications, renames and shadowing; let/fn are
# macros in core.clj even though their public docs call them special forms.
CORE_SCOPE_MACROS = dict(CORE_BINDING_MACROS)
for _name in ('let', 'loop', 'fn', 'letfn', 'case', 'comment'):
    for _prefix in ('', 'clojure.core/', 'cljs.core/'):
        CORE_SCOPE_MACROS[_prefix + _name] = _name
OPAQUE_BINDING_MACROS = OPAQUE_BINDING_MACROS | frozenset(CORE_BINDING_MACROS) | frozenset(
    'cljs.core/' + name for name in ('for', 'doseq', 'with-open', 'binding',
                                     'with-local-vars', 'as->', 'with-redefs'))
QUOTE_OR_COMMENT = frozenset({'quote','clojure.core/quote','comment','clojure.core/comment'})
BINDING_SPECIAL = frozenset({'let','loop','let*','loop*'})
# Compiler-recognized special forms must never be turned into imported calls,
# even if a malformed :refer spec happens to contain the same spelling.
COMPILER_SPECIALS = frozenset({
    'if','do','def','let','let*','loop','loop*','fn','fn*','letfn',
    'recur','try','catch','finally','throw','var','quote','new','set!',
    '.','monitor-enter','monitor-exit','import*','case*',
})


def _s(f: 'Form | None') -> str:
    return f.value if f is not None and f.kind == 'atom' else ''


def _identifier(name: str) -> bool:
    return bool(name and not name.startswith(':') and '/' not in name and
                not name[0].isdigit() and
                name not in ('&','nil','true','false'))


def binding_names(pattern: 'Form') -> frozenset[str] | None:
    """Names defined by validated simple Clojure destructuring forms.

    None means unsupported shape. Never infer locals from arbitrary expressions.
    """
    if pattern.kind == 'atom':
        return frozenset({pattern.value}) if _identifier(pattern.value) else None
    if pattern.kind == 'vector':
        # Official sequential binding grammar: [binding* & rest? :as whole?].
        # `&` and `:as` are each permitted once, and `:as` may follow `&`.
        # The former code incorrectly rejected `[x & rest :as whole]`.
        names=set()
        parts=pattern.children
        i=0
        while i < len(parts):
            val=_s(parts[i])
            if val=='&':
                if i+1>=len(parts):return None
                result=binding_names(parts[i+1])
                if result is None:return None
                names.update(result)
                i+=2
                if i<len(parts):
                    if (_s(parts[i])!=':as' or i+2!=len(parts)
                            or parts[i+1].kind!='atom'):
                        return None
                    whole=binding_names(parts[i+1])
                    if whole is None:return None
                    names.update(whole)
                return frozenset(names)
            if val==':as':
                # :as must be final and binds exactly one symbol.
                if i+2!=len(parts) or parts[i+1].kind!='atom':
                    return None
                whole=binding_names(parts[i+1])
                if whole is None:return None
                names.update(whole)
                return frozenset(names)
            result=binding_names(parts[i])
            if result is None:return None
            names.update(result)
            i+=1
        return frozenset(names)
    if pattern.kind == 'map':
        parts=pattern.children
        if len(parts)%2:return None
        names=set()
        for i in range(0,len(parts),2):
            key=_s(parts[i]); val=parts[i+1]
            # :keys / :strs / :syms, including namespace-prefixed names.
            suffix=key.lstrip(':').split('/')[-1] if key.startswith(':') else ''
            if suffix in ('keys','strs','syms'):
                if val.kind != 'vector':return None
                for child in val.children:
                    symbol=_s(child)
                    local=symbol.rsplit('/',1)[-1]
                    if not _identifier(local):return None
                    names.add(local)
            elif key == ':as':
                if val.kind!='atom':return None
                result=binding_names(val)
                if result is None:return None
                names.update(result)
            elif key == ':or':
                # Default-value expressions may execute, but do not bind names.
                if val.kind != 'map':return None
            elif key in (':select', ':all'):
                # Introduced by newer Clojure versions; no 1.12 semantics asserted.
                return None
            else:
                # Map explicit destructuring: {local-name :database/key}.
                result=binding_names(parts[i])
                if result is None:return None
                names.update(result)
        return frozenset(names)
    return None


def _has_map_default_values(form:'Form') -> bool:
    if form.kind=='map':
        items=form.children
        if any(_s(items[i])==':or' for i in range(0,len(items),2)):
            return True
    return any(_has_map_default_values(child) for child in form.children)


def _default_expression_forms(pattern: 'Form', depth: int = 0) -> tuple['Form', ...] | None:
    """Extract source expression positions from literal map :or directives.

    Unknown or malformed directives cannot justify partial call evidence.
    The resulting expressions are *possibly* evaluated on missing values,
    never proof they execute for any particular runtime collection. A default
    can see previously created destructured locals; callers conservatively
    shadow every name in the complete binding pattern to avoid linking a
    sibling binding to an unrelated global Var.
    """
    if depth > 50:
        return None
    if pattern.kind == 'atom':
        return ()
    if pattern.kind not in ('map', 'vector'):
        return None
    values=[]
    if pattern.kind=='map':
        if len(pattern.children)%2:
            return None
        # Only locals bound directly by THIS map can use its :or entries.
        # Nested patterns own their own :or; including their names here would
        # fabricate a default call ignored by clojure.core/destructure.
        direct_names=set()
        for key,value in zip(pattern.children[::2],pattern.children[1::2]):
            spelling=_s(key)
            suffix=spelling.lstrip(':').split('/')[-1] if spelling.startswith(':') else ''
            if suffix in ('keys','strs','syms') and value.kind=='vector':
                for entry in value.children:
                    local=_s(entry).rsplit('/',1)[-1]
                    if _identifier(local):
                        direct_names.add(local)
            elif spelling==':as' and value.kind=='atom' and _identifier(value.value):
                direct_names.add(value.value)
            elif key.kind=='atom' and _identifier(spelling):
                direct_names.add(spelling)
        for index in range(0,len(pattern.children),2):
            key,value=pattern.children[index:index+2]
            spelling=_s(key)
            if spelling==':or':
                if value.kind!='map' or len(value.children)%2:
                    return None
                # Clojure 1.12 evaluates defaults only for keys that are
                # actually mapped to destructured locals. A stray :or entry
                # is ignored by clojure.core/destructure, even when its value
                # looks like a function call. Later Clojure editions expand
                # the :or key grammar; this adapter does not assert their
                # semantics in the 1.12 evidence path.
                if binding_names(pattern) is None:
                    return None
                for default_key,default_expr in zip(value.children[::2],
                                                     value.children[1::2]):
                    if default_key.kind=='atom' and default_key.value in direct_names:
                        values.append(default_expr)
            elif spelling==':as' or spelling.lstrip(':').split('/')[-1] in (
                    'keys','strs','syms'):
                continue
            else:
                nested=_default_expression_forms(key,depth+1)
                if nested is None:
                    return None
                values.extend(nested)
    else:
        for item in pattern.children:
            if _s(item) in ('&',':as'):
                continue
            nested=_default_expression_forms(item,depth+1)
            if nested is None:
                return None
            values.extend(nested)
    return tuple(values)


@dataclass(frozen=True)
class ScopedCall:
    head: 'Form'
    local_name: str
    reference_form: str = 'ordinary_symbol'
    argument_count: int | None = None  # number of literal source call arguments, not evaluated values


@dataclass(frozen=True)
class ScopeSuppression:
    site: 'Form'
    name: str
    reason: str


def _fn_arities(items: tuple['Form', ...]):
    """Return (arglist, expressions) for safe fn/defn forms or None."""
    if not items:return None
    if items[0].kind == 'vector':
        arities=[(items[0],items[1:])]
    else:
        arities=[]
        for item in items:
            if item.kind!='list' or not item.children or item.children[0].kind!='vector':
                return None
            arities.append((item.children[0],item.children[1:]))
    fixed=set(); variadic=None
    for params, _ in arities:
        seen=set(); required=0; rest=False
        parts=params.children
        for i, part in enumerate(parts):
            if _s(part)=='&':
                if rest or i+2!=len(parts):return None
                rest=True
                continue
            names=binding_names(part)
            if names is None or seen.intersection(names):return None
            seen.update(names)
            if not rest:required+=1
        if rest:
            if variadic is not None:return None
            variadic=required
        else:
            if required in fixed:return None
            fixed.add(required)
    if variadic is not None and any(n>variadic for n in fixed):return None
    return arities


def referred_call_sites(body: tuple['Form', ...], referred: frozenset[str],
                        top_level_name: str | None = None, *,
                        include_qualified: bool = False,
                        include_values: bool = False,
                        include_call_heads: bool = True,
                        core_shadowed: frozenset[str] = frozenset(),
                        macro_aliases: dict[str,str | None] | None = None,
                        implicit_bindings: frozenset[str] = frozenset()) -> tuple[list[ScopedCall], list[ScopeSuppression]]:
    """Return statically recognizable *possible* unqualified call sites.

    Scope is followed for fn params, let/loop sequential bindings, letfn,
    catch, and the standard core conditional/resource/counting bindings.
    core_shadowed supplies namespace-level core exclusions/overrides known by
    the caller. Unknown binding macros/destructuring stay explicitly omitted.
    Candidate sites describe syntax, not branch reachability or macro expansion.
    """
    hits:list[ScopedCall]=[]
    suppressed:list[ScopeSuppression]=[]
    if not referred and not include_qualified:
        return hits,suppressed
    root_env=implicit_bindings | (frozenset({top_level_name}) if top_level_name else frozenset())

    def opaque(f: 'Form', reason: str):
        # Bounded output: one warning per form, not every nested token.
        suppressed.append(ScopeSuppression(f, _s(f.children[0]) if f.children else f.value,reason))

    def scan_destructuring_defaults(pattern:'Form', env:frozenset[str], depth:int):
        if not _has_map_default_values(pattern):
            return
        names=binding_names(pattern)
        expressions=_default_expression_forms(pattern)
        if names is None or expressions is None:
            opaque(pattern,'destructuring_defaults_shape_unverified')
            return
        # Names from the same pattern MAY be visible in generated default
        # binding code. Shadow all of them, even when an earlier default would
        # actually have resolved an outer Var: false negatives are preferable
        # to fabricated edges. Qualified references remain candidates.
        scan(expressions,env|names,depth+1)

    def scan(items: tuple['Form', ...], env:frozenset[str], depth:int):
        for f in items:visit(f,env,depth)

    def visit(f:'Form', env:frozenset[str], depth:int):
        if depth>70:
            opaque(f,'lexical_scope_depth_limit');return
        if f.kind=='opaque' and f.value=='#(':
            # Clojure's reader expands #(head arguments...) into a fn with
            # implicit % / %N / %& lexical parameters. Analyze that body as
            # DEFERRED syntax, never as an immediate invocation. Nested #()
            # is invalid in the reference reader and must remain opaque.
            pending=list(f.children)
            nested=False
            while pending:
                item=pending.pop()
                if item.kind=='opaque' and item.value=='#(':
                    nested=True;break
                pending.extend(item.children)
            if nested or not f.children:
                opaque(f,'nested_or_empty_anonymous_function_literal_not_resolved')
                return
            from .clojure_forms import Form
            # Any positive-index placeholder is a local argument, not a Var,
            # and Clojure permits indices beyond the usual first few.
            import re
            observed=[]
            pending=list(f.children)
            while pending:
                item=pending.pop()
                if item.kind=='atom' and re.fullmatch(r'%(?:[1-9][0-9]*|&)?',item.value):
                    observed.append(item.value)
                pending.extend(item.children)
            placeholders=frozenset({'%','%&',*observed})
            body=Form('list','(',f.children,f.line,f.col,(),f.end_line,f.end_col)
            visit(body,env|placeholders,depth+1)
            return
        if f.kind=='opaque':
            # The #' reader macro has exactly the semantics of (var symbol)
            # for a literal symbol. Quote/syntax-quote, #() and tagged forms
            # must not be interpreted as ordinary evaluated expressions.
            if include_values and f.value=="#'":
                # Reader #'sym is the compiler special form (var sym).
                # It resolves the global Var even when a local with the same
                # name is in scope.  Never process the operand as an evaluated
                # local symbol or a regular function-call argument.
                if len(f.children)==1 and f.children[0].kind=='atom':
                    symbol=f.children[0]
                    name=symbol.value
                    if name in referred or (include_qualified and '/' in name and
                                            not name.startswith(('/', ':'))):
                        hits.append(ScopedCall(symbol,name,'var_quote'))
                else:
                    opaque(f,'invalid_var_quote_literal')
            elif include_values and f.value in {'#(','`','#?','#?@'}:
                opaque(f,'reader_macro_or_conditional_value_scope_not_resolved')
            return
        if f.kind=='atom' and include_values:
            name=f.value
            if name in referred and name not in COMPILER_SPECIALS:
                if name in env:
                    suppressed.append(ScopeSuppression(f,name,'lexical_binding_shadows_refer'))
                else:
                    hits.append(ScopedCall(f,name))
            elif (include_qualified and '/' in name and
                  not name.startswith(('/', ':'))):
                hits.append(ScopedCall(f,name))
            return
        if f.kind!='list' or not f.children:
            # Collections are evaluated in expression context, but vectors of
            # bindings are handled exclusively in their special-form branch.
            scan(f.children,env,depth+1)
            return
        internal_function=f.children[0].kind=='internal_function_scope'
        head='fn*' if internal_function else _s(f.children[0])
        rest=f.children[1:]
        if head in ('quote','clojure.core/quote'):return
        if include_values and head=='var':
            # (var symbol) is a compiler special form, not normal argument
            # evaluation.  A local x does not obscure the global Var in this
            # form (unlike an ordinary use of x).  Strict shape is essential:
            # malformed forms must not emit a misleading target candidate.
            if len(rest)!=1 or rest[0].kind!='atom':
                opaque(f,'invalid_var_special_form_operand');return
            symbol=rest[0]
            name=symbol.value
            if name in referred or (include_qualified and '/' in name and
                                    not name.startswith(('/', ':'))):
                hits.append(ScopedCall(symbol,name,'var_special_form'))
            return
        macro_head=head if internal_function else (macro_aliases or {}).get(head,head)
        if macro_head is None:
            opaque(f,'ambiguous_core_macro_alias');return
        scope_head='fn' if internal_function else CORE_SCOPE_MACROS.get(macro_head,head)
        if macro_head in CORE_SCOPE_MACROS:
            if head in env or ('/' not in head and
                               (head in referred or head in core_shadowed)):
                opaque(f,'ambiguous_or_invalid_case' if scope_head=='case'
                       else 'ambiguous_core_binding_macro');return
            if scope_head=='comment':return
        if macro_head in CORE_BINDING_MACROS:
            # A local or explicitly referred name can replace an unqualified
            # core macro. Do not infer its argument evaluation or bindings.
            if head in env or ('/' not in head and
                               (head in referred or head in core_shadowed)):
                opaque(f,'ambiguous_core_binding_macro');return
            macro=CORE_BINDING_MACROS[macro_head]
            if macro=='as->':
                if len(rest)<2 or rest[1].kind!='atom' or binding_names(rest[1]) is None:
                    opaque(f,'invalid_or_unsupported_core_binding_macro');return
                visit(rest[0],env,depth+1)
                scan(rest[2:],env|binding_names(rest[1]),depth+1)
                return
            if macro in ('for','doseq'):
                if (not rest or rest[0].kind!='vector' or
                        (macro=='for' and (len(rest)!=2 or not rest[0].children))):
                    opaque(f,'invalid_or_unsupported_core_binding_macro');return
                parts=rest[0].children
                if len(parts)%2:
                    opaque(f,'invalid_or_unsupported_core_binding_macro');return
                # Validate all clauses before visiting anything. The plan
                # preserves lexical order but claims neither lazy realization
                # nor runtime filtering/iteration counts.
                plan=[]
                seen_generator=False
                for i in range(0,len(parts),2):
                    pattern,expr=parts[i:i+2]
                    marker=_s(pattern)
                    if marker.startswith(':'):
                        if not seen_generator or marker not in (':let',':when',':while'):
                            opaque(f,'invalid_or_unsupported_core_binding_macro');return
                        if marker==':let':
                            if expr.kind!='vector' or len(expr.children)%2:
                                opaque(f,'invalid_or_unsupported_core_binding_macro');return
                            for j in range(0,len(expr.children),2):
                                binding,value=expr.children[j:j+2]
                                names=binding_names(binding)
                                if names is None:
                                    opaque(f,'invalid_or_unsupported_core_binding_macro');return
                                plan.append((binding,value,names))
                        else:
                            plan.append((None,expr,frozenset()))
                    else:
                        names=binding_names(pattern)
                        if names is None:
                            opaque(f,'invalid_or_unsupported_core_binding_macro');return
                        seen_generator=True
                        plan.append((pattern,expr,names))
                local=env
                for pattern,expr,names in plan:
                    visit(expr,local,depth+1)
                    if pattern is not None:
                        scan_destructuring_defaults(pattern,local,depth+1)
                    local=local|names
                scan(rest[1:],local,depth+1)
                return
            conditional=macro in ('if-let','if-some')
            if (not rest or rest[0].kind!='vector' or
                    (conditional and len(rest) not in (2,3))):
                opaque(f,'invalid_or_unsupported_core_binding_macro');return
            pairs=rest[0].children
            if ((macro!='with-open' and len(pairs)!=2) or len(pairs)%2):
                opaque(f,'invalid_or_unsupported_core_binding_macro');return
            # Validate the entire binding shape before collecting any sites;
            # malformed forms cannot publish partial candidate evidence.
            bindings=[]
            for i in range(0,len(pairs),2):
                pattern=pairs[i]
                new=binding_names(pattern)
                if (new is None or (macro in ('dotimes','with-open') and
                                    pattern.kind!='atom')):
                    opaque(f,'invalid_or_unsupported_core_binding_macro');return
                bindings.append(new)
            local=env
            for index,new in enumerate(bindings):
                pattern=pairs[index*2]
                visit(pairs[index*2+1],local,depth+1)
                scan_destructuring_defaults(pattern,local,depth+1)
                local=local|new
            if conditional:
                visit(rest[1],local,depth+1)
                if len(rest)==3:visit(rest[2],env,depth+1)
            else:
                scan(rest[1:],local,depth+1)
            return
        if head in OPAQUE_BINDING_MACROS:
            opaque(f,'unverified_binding_macro_or_literal_case');return
        if scope_head=='case':
            # Clojure's case macro treats test constants as reader data; only
            # the dispatch expression and matching results are evaluated.
            # If `case` is locally shadowed, don't guess macro semantics.
            if head in env or not rest:
                opaque(f,'ambiguous_or_invalid_case');return
            visit(rest[0],env,depth+1)
            for index in range(2,len(rest),2):
                visit(rest[index],env,depth+1)
            if len(rest)%2==0:
                visit(rest[-1],env,depth+1)
            return
        if scope_head in BINDING_SPECIAL:
            if not rest or rest[0].kind!='vector' or len(rest[0].children)%2:
                opaque(f,'invalid_or_unhandled_sequential_bindings');return
            local=env
            pairs=rest[0].children
            # Validate all patterns first; an invalid later binding must not
            # leave calls from earlier initializers in the evidence graph.
            if any(binding_names(pairs[i]) is None for i in range(0,len(pairs),2)):
                opaque(f,'unsupported_destructuring');return
            for i in range(0,len(pairs),2):
                visit(pairs[i+1],local,depth+1)
                new=binding_names(pairs[i])
                # :or expressions in map destructuring are compiler-generated
                # defaults. We currently don't model their evaluation order;
                # record this omission rather than imply there was no call.
                scan_destructuring_defaults(pairs[i],local,depth+1)
                if new is None:
                    opaque(f,'unsupported_destructuring');return
                local=local|new
            scan(rest[1:],local,depth+1)
            return
        if scope_head in ('fn','fn*'):
            i=0;scope=env
            if rest and _identifier(_s(rest[0])):
                scope=scope|{_s(rest[0])};i=1
            arities=_fn_arities(rest[i:])
            if arities is None:
                opaque(f,'unsupported_fn_arity');return
            from .clojure_function_scope import function_contract_plan
            plans=[function_contract_plan(params,exprs,expand_contracts=scope_head=='fn')
                   for params,exprs in arities]
            if any(plan is None for plan in plans):
                opaque(f,'unsupported_function_conditions');return
            for (params, exprs),(pre,body,post) in zip(arities,plans):
                names=binding_names(params)
                if names is None:
                    opaque(f,'unsupported_fn_destructuring');return
                scan_destructuring_defaults(params,scope,depth+1)
                scan(pre,scope|names,depth+1)
                scan(body,scope|names,depth+1)
                scan(post,scope|names|{'%'},depth+1)
            return
        if scope_head=='letfn':
            if not rest or rest[0].kind!='vector':
                opaque(f,'unsupported_letfn');return
            definitions=rest[0].children
            names=[]
            for spec in definitions:
                if spec.kind!='list' or len(spec.children)<2 or not _identifier(_s(spec.children[0])):
                    opaque(f,'unsupported_letfn');return
                names.append(_s(spec.children[0]))
            if len(set(names))!=len(names):
                opaque(f,'unsupported_letfn');return
            all_arities=[_fn_arities(spec.children[1:]) for spec in definitions]
            if any(arities is None for arities in all_arities):
                opaque(f,'unsupported_letfn_arity');return
            from .clojure_function_scope import function_contract_plan
            all_plans=[[function_contract_plan(params,exprs) for params,exprs in arities]
                       for arities in all_arities]
            if any(plan is None for plans in all_plans for plan in plans):
                opaque(f,'unsupported_function_conditions');return
            scope=env|frozenset(names)
            for arities,plans in zip(all_arities,all_plans):
                for (params,exprs),(pre,body,post) in zip(arities,plans):
                    binds=binding_names(params)
                    if binds is None:
                        opaque(f,'unsupported_letfn_destructuring');return
                    scan_destructuring_defaults(params,scope,depth+1)
                    scan(pre,scope|binds,depth+1)
                    scan(body,scope|binds,depth+1)
                    scan(post,scope|binds|{'%'},depth+1)
            scan(rest[1:],scope,depth+1)
            return
        if head=='catch':
            if len(rest)<3 or not _identifier(_s(rest[1])):
                opaque(f,'unsupported_catch_binding');return
            scan(rest[2:],env|{_s(rest[1])},depth+1)
            return
        if include_call_heads and head in referred and head not in COMPILER_SPECIALS:
            if head in env:
                suppressed.append(ScopeSuppression(f.children[0],head,'lexical_binding_shadows_refer'))
            else:
                hits.append(ScopedCall(f.children[0],head,argument_count=len(rest)))
        if (include_call_heads and include_qualified and '/' in head and not head.startswith(('/', ':'))
                and head not in QUOTE_OR_COMMENT):
            # Only executable call-head positions in scopes the scanner can
            # model. Macro arguments may still be data: candidate, never exact.
            hits.append(ScopedCall(f.children[0],head,argument_count=len(rest)))
        if not head:
            # Computed function position is itself an expression, e.g. an
            # immediately invoked fn. Its bindings do not cover the arguments.
            visit(f.children[0],env,depth+1)
        scan(rest,env,depth+1)
    scan(body,root_env,0)
    return hits,suppressed
