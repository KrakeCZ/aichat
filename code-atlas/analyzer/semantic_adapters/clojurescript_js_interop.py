"""Offline ClojureScript JavaScript interop syntax evidence.

No JavaScript engine, Closure compiler, npm modules, project macros or
application code are executed. `js/` names and `.`/`.-` forms are source
expressions, NOT declarations proven to exist in a target JS environment.
ClojureScript .cljc platforms cannot be inferred from the filename alone.
"""
from __future__ import annotations

import re
from .clojure_forms import Form, _definition
from .clojure_scope import referred_call_sites

MAX_JS_INTEROP_SITES = 1000
_JS_SYMBOL = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*\Z')
_JS_MEMBER = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*\Z')


def analyze_cljs_js_interop(forms, *, aliases, alias_guards, shadowed_core,
                            macro_aliases):
    """Observe syntax in executable expression positions, never infer JS targets."""
    sites=[]
    seen=set()
    suppressed=0
    truncated=False
    namespace_js_ambiguous = 'js' in aliases or 'js' in alias_guards

    def append(owner, form, kind, *, symbol=None, member=None, argc=None, phase=None):
        nonlocal truncated
        if len(sites)>=MAX_JS_INTEROP_SITES:
            truncated=True
            return
        key=(owner,form.line,form.col,kind,member,symbol)
        if key in seen:return
        seen.add(key)
        sites.append({
            'owner':owner, 'kind':kind,'symbol':symbol,'member':member,
            'supplied_argument_count':argc,
            'line':form.line,'column':form.col,
            'end_line':form.end_line,'end_column':form.end_col,
            'source_platform':'cljs','analysis_phase':phase,
            'resolution':'clojurescript_js_interop_source_syntax_only',
            'javascript_binding_confirmed':False,
            'runtime_property_exists_confirmed':False,
            'closure_renaming_or_externs_checked':False,
        })

    def build_list_index(forms):
        by_head={}
        stack=list(forms)
        while stack:
            form=stack.pop()
            if form.kind=='opaque' and form.value!='#(':
                continue
            if form.kind=='list' and form.children:
                head=form.children[0]
                if head.kind=='atom':
                    by_head[(head.line,head.col)]=form
            stack.extend(form.children)
        return by_head

    for top in forms:
        info=_definition(top)
        if not info or info[1] not in {'function','macro','variable'}:
            continue
        owner,kind,_=info
        body=list(top.children[2:])
        if kind in {'function','macro'}:
            if body and body[0].kind=='string':body.pop(0)
            if body and body[0].kind=='map':body.pop(0)
            if body and body[0].kind=='list' and body[-1].kind=='map':body.pop()
            marker=Form('internal_function_scope','fn*',(),top.line,top.col)
            body=[Form('list','(',tuple([marker,*body]),top.line,top.col)]
        # Scope-aware scanner handles fn, let, letfn, catch, quote, core binding
        # macros and anonymous shorthand with its own fail-closed suppression.
        hits_call,ignored_call=referred_call_sites(tuple(body),frozenset(),
            include_qualified=True,core_shadowed=shadowed_core,
            macro_aliases=macro_aliases)
        hits_value,ignored_value=referred_call_sites(tuple(body),frozenset(),
            include_qualified=True,include_values=True,
            include_call_heads=False,core_shadowed=shadowed_core,
            macro_aliases=macro_aliases)
        suppressed+=len(ignored_call)+len(ignored_value)
        phase=('macro_expansion_source_candidate' if kind=='macro'
               else 'unexpanded_expression_candidate')
        # When an explicit alias conflicts with `js`, suppress only the
        # global-name observation. Dot interop on local receivers remains
        # independent of that namespace ambiguity.
        heads_index=build_list_index(body)
        for hit in hits_call:
            literal=hit.head.value
            if namespace_js_ambiguous or not literal.startswith('js/'):
                continue
            value=literal[3:]
            if not _JS_SYMBOL.fullmatch(value.rstrip('.')):
                continue
            form=heads_index.get((hit.head.line,hit.head.col))
            if not form:continue
            if value.endswith('.'):
                tag='js_global_constructor_candidate'
            else:
                tag='js_global_function_or_member_call_candidate'
            append(owner,hit.head,tag,symbol=literal,
                   argc=len(form.children)-1,phase=phase)
        for hit in hits_value:
            literal=hit.head.value
            if namespace_js_ambiguous or not literal.startswith('js/'):
                continue
            value=literal[3:]
            if not _JS_SYMBOL.fullmatch(value):
                continue
            append(owner,hit.head,'js_global_value_reference_candidate',
                   symbol=literal,phase=phase)
        # The dot forms can be recognized syntactically, but their receiver
        # type is unknown. Restrict the traversal to literal expression trees,
        # skipping quoted data and reader macros. Never make a call graph edge.
        def visit(form,depth=0):
            if depth>60:return
            if form.kind=='opaque':
                if form.value=='#(' and form.children:
                    pending=list(form.children)
                    while pending:
                        child=pending.pop()
                        if child.kind=='opaque' and child.value=='#(':
                            return  # invalid nested reader function
                        pending.extend(child.children)
                    visit(Form('list','(',form.children,form.line,form.col),depth+1)
                return
            if form.kind!='list':
                for child in form.children:visit(child,depth+1)
                return
            if not form.children:return
            head=form.children[0]
            h=head.value if head.kind=='atom' else ''
            if h in ('quote','clojure.core/quote','comment','clojure.core/comment'):
                return
            from .clojure_scope import OPAQUE_BINDING_MACROS
            if h in OPAQUE_BINDING_MACROS:
                return
            # `case` tests are literal data, not evaluated expressions.
            if h in ('case','clojure.core/case'):
                if len(form.children)>1:visit(form.children[1],depth+1)
                for index in range(3,len(form.children),2):
                    visit(form.children[index],depth+1)
                if len(form.children)%2==1 and len(form.children)>2:
                    visit(form.children[-1],depth+1)
                return
            if h.startswith('.-') and _JS_MEMBER.fullmatch(h[2:]) and len(form.children)==2:
                append(owner,head,'js_instance_property_access_syntax',member=h[2:],phase=phase)
            elif h.startswith('.') and _JS_MEMBER.fullmatch(h[1:]) and len(form.children)>=2:
                append(owner,head,'js_instance_method_call_syntax',member=h[1:],
                       argc=len(form.children)-2,phase=phase)
            elif h=='.' and len(form.children)>=3:
                member=form.children[2]
                if member.kind=='atom' and member.value.startswith('-') and _JS_MEMBER.fullmatch(member.value[1:]):
                    append(owner,member,'js_dot_property_access_syntax',member=member.value[1:],phase=phase)
                elif member.kind=='atom' and _JS_MEMBER.fullmatch(member.value):
                    append(owner,member,'js_dot_method_or_property_syntax',member=member.value,
                           argc=len(form.children)-3,phase=phase)
            for child in form.children[1:]:visit(child,depth+1)
        for form in body:visit(form)
    return {'status':('bounded_evidence_unavailable_site_limit' if truncated
                      else 'cljs_js_source_syntax_not_compiler_or_runtime_resolution'),
            'site_count':len(sites) if not truncated else 0,
            'sites':sites if not truncated else [],
            'scope_suppressions':suppressed,
            'js_alias_ambiguous':namespace_js_ambiguous,
            'limits':{'max_sites':MAX_JS_INTEROP_SITES},
            'limitations':['not_js_property_existence_evidence',
                'not_closure_compiler_externs_or_renaming',
                'not_npm_module_resolution','not_macro_expanded',
                'no_js_runtime_type_or_dispatch_assertions']}
