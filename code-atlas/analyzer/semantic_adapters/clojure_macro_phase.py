"""Offline candidate evidence for ClojureScript :require-macros compiler phase.

A *separate* macro compilation phase is not a runtime dependency. This module
cannot attest ClojureScript compiler, classpath, compiler options, or actual
macro expansion. Explicit clauses and bounded, source-evidenced implicit self-macro loading are captured.
"""
from __future__ import annotations

from pathlib import Path

MAX_DECLARATIONS = 250
MAX_UNRESOLVED = 80
MAX_SYMBOL_EVIDENCE = 500
MAX_PROVIDER_DECLARATIONS = 500


def _use_macro_only_libspec(spec) -> tuple[list[str], str | None]:
    """Exact evidence subset for `:use-macros [lib :only [symbols]]`.

    The ClojureScript ns guide documents :use-macros / :only as the dual of
    :require-macros / :refer. More complex use/rename semantics are *not*
    inferred here. This validates data shapes, not actual compiler exports.
    """
    from .clojure_forms import _sym
    if spec.kind != 'vector' or len(spec.children) < 2:
        return [], 'use_macros_requires_explicit_only_vector'
    namespace=_sym(spec.children[0])
    if not namespace or namespace.startswith(':') or '/' in namespace:
        return [], 'invalid_macro_namespace'
    options=spec.children[1:]
    if len(options)%2:
        return [], 'missing_use_macros_option_value'
    if len(options)!=2 or _sym(options[0])!=':only':
        return [], 'use_macros_options_outside_verified_only_subset'
    value=options[1]
    if value.kind!='vector' or not value.children:
        return [], 'use_macros_only_requires_nonempty_vector'
    names=[_sym(v) for v in value.children]
    if any(not n or n.startswith(':') or '/' in n for n in names):
        return [], 'invalid_use_macros_only_symbol'
    if len(names)!=len(set(names)):
        return [], 'duplicate_use_macros_only_symbol'
    return names, None


def macro_require_specs(forms: list, *, platform: str, validation) -> dict:
    """Collect explicit literal CLJS macro imports; do NOT execute ns.

    This records :require-macros, an explicit verified-shape :use-macros
    [lib :only [syms]], and the documented inline :require sugar,
    :include-macros true and :refer-macros [syms]. These are *requested*
    names, not verified exports/calls. A real ClojureScript compiler and
    classpath are still needed to establish resolution. Implicit macro loading
    is handled separately by the bounded implicit-macro evidence module.
    """
    from .clojure_forms import _sym, _top
    if platform != 'cljs':
        return {'status':'not_applicable','requires':[],'limitations':[]}
    ns=[f for f in forms if _top(f,'ns')]
    if len(ns)!=1:
        return {'status':'unavailable','requires':[],
                'limitations':['missing_or_ambiguous_namespace']}
    out=[];limitations=[];overflow=0

    def note(spec,reason):
        nonlocal overflow
        if len(limitations)<MAX_UNRESOLVED:
            limitations.append({'line':spec.line,'column':spec.col,
                                'reason':reason,'phase':'compile_time_macro'})
        else:overflow+=1

    def include(spec,origin,requested=(),alias=None,requested_forms=()):
        nonlocal overflow
        if len(out)>=MAX_DECLARATIONS:
            overflow+=1;return
        namespace=_sym(spec) if spec.kind=='atom' else _sym(spec.children[0])
        out.append({'namespace':namespace,'line':spec.line,'column':spec.col,
                    'phase':'compile_time_macro','origin':origin,
                    'requested_refer_macros':list(requested),
                    'requested_macro_sites':[{'name':name,'line':site.line,'column':site.col}
                       for name,site in zip(requested,requested_forms)],
                    'requested_alias':alias})

    for clause in ns[0].children[2:]:
        if clause.kind != 'list' or not clause.children:
            continue
        head=_sym(clause.children[0])
        if head not in {':require',':require-macros',':use-macros'}:
            continue
        for spec in clause.children[1:]:
            if head==':use-macros':
                only_names, issue=_use_macro_only_libspec(spec)
                if issue:
                    note(spec,issue);continue
                include(spec,':use-macros/:only',only_names,requested_forms=spec.children[-1].children)
                continue
            if head==':require-macros':
                issue=validation(spec,'clj')
                if issue:
                    note(spec,issue);continue
                # :as-alias never loads a library. It is not an explicit
                # compile-time loading declaration.
                if spec.kind=='vector' and any(_sym(part)==':as-alias'
                                               for part in spec.children[1:]):
                    continue
                # `:refer` and `:as` are intent, *not* proof that the
                # requested macro symbols exist in the indexed namespace.
                options=spec.children[1:] if spec.kind=='vector' else ()
                aliases={_sym(options[i]):options[i+1]
                         for i in range(0,len(options)-1,2)}
                alias=_sym(aliases[':as']) if ':as' in aliases else None
                refs=([_sym(item) for item in aliases[':refer'].children]
                      if ':refer' in aliases and aliases[':refer'].kind in ('vector','list') else [])
                include(spec,':require-macros',refs,alias,
                        requested_forms=(aliases[':refer'].children if ':refer' in aliases and
                                         aliases[':refer'].kind in ('vector','list') else ()))
                continue
            # Inline sugar is recognized *only* as :require vector options.
            if spec.kind!='vector' or len(spec.children)<3:
                continue
            options=spec.children[1:]
            keys=[_sym(options[i]) for i in range(0,len(options),2)]
            if not any(x in {':include-macros',':refer-macros'} for x in keys):
                continue
            issue=validation(spec,'cljs')
            if issue:
                note(spec,issue);continue
            pairs={_sym(options[i]):options[i+1] for i in range(0,len(options),2)}
            alias=_sym(pairs[':as']) if ':as' in pairs else None
            if ':include-macros' in pairs:
                if _sym(pairs[':include-macros'])=='true':
                    include(spec,':require/:include-macros',alias=alias)
                # false means explicitly no macro loading, not unresolved.
            elif ':refer-macros' in pairs:
                requested=[_sym(item) for item in pairs[':refer-macros'].children]
                include(spec,':require/:refer-macros',requested,alias,
                        requested_forms=pairs[':refer-macros'].children)
    return {'status':'candidate_not_semantically_verified','requires':out,
            'limitations':limitations,'overflow_count':overflow}


def _direct_macro_declarations(path: str, source: dict) -> tuple[dict[str,list[dict]], bool]:
    """Collect *literal top-level* defmacro forms from this compiler platform.

    This never macroexpands, loads classpath, or invokes an input project's
    source. A match is a directly observed declaration, NOT a Var resolution.
    """
    from .clojure_forms import _definition, _callable_visibility
    forms=source['forms']
    if Path(path).suffix.lower()=='.cljc':
        from .clojure_conditionals import project, ProjectionUnavailable
        try:
            forms=project(forms,'clj').forms
        except ProjectionUnavailable:
            return {},True
    result:dict[str,list[dict]]={}
    count=0
    for form in forms:
        recognized=_definition(form)
        if not recognized or recognized[1]!='macro':
            continue
        if count>=MAX_PROVIDER_DECLARATIONS:
            return result,True
        count+=1
        name,_,part=recognized
        visibility,reason=_callable_visibility(form)
        result.setdefault(name,[]).append({
            'name':name,'visibility':visibility,
            'visibility_evidence':reason,'line':part.line,'column':part.col})
    return result,False


def _macro_symbol_evidence(req: dict, declarations: dict[str,list[dict]],
                           truncated: bool, target_path: str, sha: str,
                           source_path: str, source_sha: str) -> tuple[list[dict],int]:
    """Compare requested symbols only with explicit provider defmacro forms.

    Missing direct declarations can reflect macro generation, other source
    roots, or classpath changes, so absence must NEVER mean compiler error.
    """
    items=[]
    requested=req.get('requested_refer_macros',[])
    sites={site['name']:site for site in req.get('requested_macro_sites',[])}
    excess=max(0,len(requested)-MAX_SYMBOL_EVIDENCE)
    for name in requested[:MAX_SYMBOL_EVIDENCE]:
        matches=declarations.get(name,[])
        if truncated:
            status='provider_declaration_scan_truncated'
        elif not matches:
            status='no_literal_defmacro_found_not_proof_of_absence'
        elif len(matches)!=1:
            status='ambiguous_multiple_direct_defmacro_declarations'
        elif matches[0]['visibility']=='private':
            status='private_direct_defmacro_not_public_candidate'
        elif matches[0]['visibility']!='public':
            status='unverified_private_metadata_no_public_claim'
        else:
            status='single_public_direct_defmacro_declaration_candidate'
        site=sites.get(name,{})
        row={'symbol':name,'status':status,'compiler_verified':False,
             'source_evidence':{'path':source_path,'line':site.get('line',req['line']),
                 'column':site.get('column',req['column']),'source_sha256':source_sha},
             'matching_direct_defmacro_count':len(matches)}
        if len(matches)==1:
            m=matches[0]
            row['target_declaration_evidence']={
                'path':target_path,'line':m['line'],'column':m['column'],
                'source_sha256':sha,'declaration_kind':'defmacro',
                'visibility':m['visibility'],
                'visibility_evidence':m['visibility_evidence']}
        items.append(row)
    return items,excess


def build_macro_phase_evidence(files: dict[str,dict], namespaces: dict[str,dict]) -> dict:
    """Resolve only unique *indexed* :clj macro providers, never executable edges.

    Shared .cljc providers are interpreted only through their JVM projection.
    .cljs is not accepted as a macro provider in regular JVM-hosted CLJS. A
    self-hosted CLJS compiler needs a distinct future adapter and build proof.
    """
    provider_index={}
    provider_defmacros={}
    for path, source in sorted(files.items()):
        if Path(path).suffix.lower() not in ('.clj','.cljc'):
            continue
        provider=namespaces.get(path,{}).get('clj',{})
        if provider.get('status')!='candidate_not_semantically_verified':
            continue
        ns=provider.get('namespace')
        if isinstance(ns,str) and ns:
            provider_index.setdefault(ns,[]).append((path,source['sha'],provider))
            provider_defmacros[path]=_direct_macro_declarations(path,source)
    runtime_index={}
    for runtime_path, views in namespaces.items():
        view=views.get('cljs',{})
        if view.get('status')=='candidate_not_semantically_verified':
            runtime_index.setdefault(view['namespace'],[]).append(runtime_path)
    from .clojure_implicit_macros import implicit_macro_specs
    from .clojure_source_context import select_namespace_paths, source_path_evidence
    out={}
    for path,source in sorted(files.items()):
        declared=source.get('macro_require_specs',{})
        if Path(path).suffix.lower() not in ('.cljs','.cljc'):
            continue
        rows=[];unresolved=[];overflow=int(declared.get('overflow_count',0))
        symbol_evidence_count=0
        unresolved_symbol_evidence_count=0
        context=source.get('project_source_context',{})
        def selected_runtime(name):
            return select_namespace_paths(context,name,'cljs',runtime_index[name])[0]
        def selected_providers(name):
            chosen=select_namespace_paths(context,name,'clj',[p for p,_,_ in provider_index[name]])[0]
            return [row for row in provider_index[name] if row[0] in set(chosen)]
        scoped_runtime={name:selected_runtime(name) for name in runtime_index}
        scoped_providers={name:selected_providers(name) for name in provider_index}
        implicit, implicit_unresolved, implicit_overflow=implicit_macro_specs(
            path,source,files,namespaces,scoped_runtime,scoped_providers,provider_defmacros)
        unresolved.extend(implicit_unresolved)
        overflow+=implicit_overflow
        for req in [*declared.get('requires',[]),*implicit]:
            name=req['namespace'];targets=scoped_providers.get(name,[])
            _,reason=select_namespace_paths(context,name,'clj',[p for p,_,_ in provider_index.get(name,[])])
            if not reason and not targets:reason='macro_namespace_absent_from_indexed_clj_sources'
            elif not reason and len(targets)!=1:reason='ambiguous_macro_namespace_providers'
            elif not reason and targets[0][0]==path:reason='self_namespace_macro_requires_build_phase_proof'
            if reason:
                if len(unresolved)<MAX_UNRESOLVED:
                    unresolved.append({'namespace':name,'reason':reason,
                                       'line':req['line'],'column':req['column'],
                                       'candidate_provider_count':len(targets)})
                else:overflow+=1
                continue
            target_path,sha,ns_record=targets[0]
            if len(rows)>=MAX_DECLARATIONS:
                overflow+=1;continue
            declarations,scan_truncated=provider_defmacros.get(target_path,({},True))
            symbol_evidence,symbol_excess=_macro_symbol_evidence(
                req,declarations,scan_truncated,target_path,sha,path,source['sha'])
            overflow+=symbol_excess
            symbol_evidence_count+=len(symbol_evidence)
            unresolved_symbol_evidence_count+=sum(
                e['status']!='single_public_direct_defmacro_declaration_candidate'
                for e in symbol_evidence)
            rows.append({'relation':'cljs_compile_time_macro_require_candidate',
                        'phase':'compile_time_macro',
                        'source_platform':'cljs','provider_platform':'clj',
                        'source_path':path,'target_path':target_path,'namespace':name,
                        'origin':req.get('origin',':require-macros'),
                        'requested_refer_macros':req.get('requested_refer_macros',[]),
                        'requested_alias':req.get('requested_alias'),
                        'macro_symbols_verified':False,
                        'direct_macro_declaration_evidence':symbol_evidence,
                        'direct_macro_declaration_scan_truncated':scan_truncated,
                        'direct_macro_declaration_evidence_truncated':symbol_excess,
                        'source_evidence':{'path':path,'line':req['line'],'column':req['column'],
                                           'source_sha256':source['sha']},
                        'target_evidence':{'path':target_path,'line':ns_record.get('namespace_line'),
                                           'column':ns_record.get('namespace_column'),
                                           'source_sha256':sha},
                        'semantic_resolution_certified':False,
                        'requires_build_classpath_confirmation':True,
                        'requires_macroexpansion_confirmation':True,
                        'resolution':('declared_project_macro_source_candidate_no_effective_basis' if context.get('status')=='project_paths_declared' else 'unique_indexed_macro_namespace_only_no_compiler_proof')})
            proof=source_path_evidence(context,name,'clj',target_path)
            if proof:rows[-1]['project_source_path_evidence']=proof
            if 'implicit_loading_evidence' in req:
                rows[-1]['implicit_loading_evidence']=req['implicit_loading_evidence']
        out[path]={
            'status':declared.get('status','unavailable'),
            'phase':'compile_time_macro',
            'compiler_executed':False,'classpath_resolved':False,
            'macro_symbol_evidence_count':symbol_evidence_count,
            'unresolved_macro_symbol_evidence_count':unresolved_symbol_evidence_count,
            'macroexpansion_certified':False,'candidate_count':len(rows),
            'unresolved_count':len(unresolved),'overflow_count':overflow,
            'candidates':rows,'unresolved':unresolved,
            'limitations':declared.get('limitations',[]),
            'excludes_implicit_macros':False,
            'implicit_macro_scope':'unique_runtime_namespace_with_explicit_self_macro_require_and_public_literal_refer',
            'excludes_inline_macro_sugar':False,
            'inline_macro_sugar_scope':'literal_include_true_false_and_refer_vector',
            'explicit_use_macros_scope':'vector_libspec_with_nonempty_only_symbols',
            'excludes_self_hosted_cljs':True,
        }
    return out
