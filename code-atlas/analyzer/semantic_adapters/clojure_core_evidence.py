"""Lexical guard and source mapping for optional trusted Clojure core Var oracle.

No macro from the analyzed project is executed.  Project namespace mutations
which are not statically proven are explicitly outside this conservative tier.
"""
from __future__ import annotations

from .clojure_core_oracle import SAFE_MACROS
from .clojure_scope import referred_call_sites, COMPILER_SPECIALS, CORE_SCOPE_MACROS
from .clojure_forms import _definition, _sym, _top, Form

MAX_SITES_PER_FILE=512


def _clojure_core_namespace_policy(forms):
    """Only a literal ns and a valid :refer-clojure subset are eligible."""
    ns=[f for f in forms if _top(f,'ns')]
    if len(ns)!=1:return None,{'reason':'missing_or_ambiguous_ns'}
    clauses=[cl for cl in ns[0].children[2:] if _top(cl,':refer-clojure')]
    if len(clauses)>1:return None,{'reason':'duplicate_refer_clojure_clause'}
    if not clauses:return (None,set()),{}
    pieces=clauses[0].children[1:]
    if len(pieces)%2:return None,{'reason':'odd_refer_clojure_options'}
    only=None;excluded=set();seen=set()
    for i in range(0,len(pieces),2):
        key,val=_sym(pieces[i]),pieces[i+1]
        if key in seen or key not in (':only',':exclude',':rename'):
            return None,{'reason':'unmodeled_refer_clojure_options'}
        seen.add(key)
        if key in (':only',':exclude'):
            if val.kind not in ('vector','list') or any(x.kind!='atom' or '/' in x.value
                                                        or x.value.startswith(':') for x in val.children):
                return None,{'reason':'invalid_refer_clojure_name_vector'}
            members={x.value for x in val.children}
            if key==':only':only=members
            else:excluded=members
        else:
            # Renaming changes the identity of local spellings; do not make
            # a default-core conclusion in the presence of :rename until
            # every conflict is statically established.
            return None,{'reason':'refer_clojure_rename_requires_compiler_env'}
    return (only,excluded),{}


def _slice_form(text,form):
    lines=text.splitlines(keepends=True)
    l,c,e,ec=form.line,form.col,form.end_line,form.end_col
    if not isinstance(e,int) or not isinstance(ec,int) or l<1 or l>len(lines) or e<l or e>len(lines):
        return None
    start=sum(len(x) for x in lines[:l-1])+c-1
    end=sum(len(x) for x in lines[:e-1])+ec-1
    if end<=start or end-start>32_000:return None
    return text[start:end]


def discover_core_sites(forms,text,refer_names):
    policy,problem=_clojure_core_namespace_policy(forms)
    if policy is None:
        return {'sites':[],'macros':[],'limitations':[problem]}
    only,excluded=policy
    definitions={r[0] for f in forms if (r:=_definition(f))}
    # Collect actual executable list heads, not every word in a string,
    # quoted data, binding vector or comment macro.
    candidate_heads=set();forms_by_position={}
    def walk(f,depth=0):
        if depth>80 or f.kind=='opaque':return
        if f.kind=='list' and f.children and f.children[0].kind=='atom':
            head=f.children[0]
            forms_by_position[(head.line,head.col)]=f
            if head.value and '/' not in head.value and head.value not in COMPILER_SPECIALS:
                candidate_heads.add(head.value)
        for ch in f.children:walk(ch,depth+1)
    for f in forms:walk(f)
    permitted={n for n in candidate_heads if (only is None or n in only)
               and n not in excluded and n not in definitions and n not in refer_names
               and n not in CORE_SCOPE_MACROS and not n.startswith(':')}
    if not permitted:return {'sites':[],'macros':[],'limitations':[]}
    from .clojure_core_context import shadowed_core_names,core_macro_aliases
    shadowed=shadowed_core_names(forms,refer_names)
    aliases=core_macro_aliases(forms)
    sites=[];macro_requests=[];limitations=[]
    for form in forms:
        parsed=_definition(form)
        if not parsed or parsed[1] not in ('function','macro'):continue
        declaration=parsed[0]
        parts=list(form.children[2:])
        if parts and parts[0].kind=='string':parts.pop(0)
        if parts and parts[0].kind=='map':parts.pop(0)
        if parts and parts[0].kind=='list' and parts[-1].kind=='map':parts.pop()
        synthetic=Form('internal_function_scope','fn*',(),form.line,form.col)
        body=(Form('list','(',tuple([synthetic,*parts]),form.line,form.col),)
        found,suppressed=referred_call_sites(body,frozenset(permitted),declaration,
                    core_shadowed=shadowed,macro_aliases=aliases,
                    implicit_bindings=frozenset({'&form','&env'}) if parsed[1]=='macro' else frozenset())
        for site in found:
            if len(sites)>=MAX_SITES_PER_FILE:
                limitations.append({'reason':'core_var_sites_truncated'});break
            item={'name':site.local_name,'containing_declaration':declaration,
                  'location':{'line':site.head.line,'column':site.head.col,
                              'end_line':site.head.end_line,'end_column':site.head.end_col},
                  'resolution':'pending_core_oracle',
                  'not_a_runtime_call_claim':True}
            sites.append(item)
            if site.local_name in SAFE_MACROS:
                original=forms_by_position.get((site.head.line,site.head.col))
                if original is not None:
                    src=_slice_form(text,original)
                    if src and src.startswith('('+site.local_name):
                        macro_requests.append((item,site.local_name,src))
        if suppressed:
            limitations.append({'reason':'lexical_candidates_suppressed', 'count':len(suppressed)})
    return {'sites':sites,'macros':macro_requests[:MAX_SITES_PER_FILE],
            'limitations':limitations}
