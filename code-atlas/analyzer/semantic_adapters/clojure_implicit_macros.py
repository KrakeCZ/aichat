"""Bounded source-only evidence for documented CLJS implicit macro loading.

The runtime namespace must explicitly require its own macro namespace. This
module never loads it, follows transitive exports, or claims compiler resolution.
"""
from pathlib import Path


def implicit_macro_specs(path, source, files, namespaces, runtime_index, provider_index,
                         provider_defmacros):
    from .clojure_forms import _sym, _top, _libspec_issue, _definition, _explicit_refers
    from .clojure_conditionals import project, ProjectionUnavailable
    from .clojure_macro_phase import MAX_DECLARATIONS, MAX_UNRESOLVED

    def platform_forms(name, record, platform):
        if Path(name).suffix.lower()=='.cljc':
            try:
                return project(record['forms'],platform).forms
            except ProjectionUnavailable:
                return ()
        return record['forms']

    def evidence(name, record, line, column):
        return {'path':name,'line':line,'column':column,'source_sha256':record['sha']}

    def literal_symbol(form):
        name=_sym(form)
        return bool(name and name not in {'true','false','nil'} and
                    not name[0].isdigit() and not name.startswith(':') and
                    not (name[0] in '+-.' and len(name)>1 and name[1].isdigit()))

    def namespace_forms(name, record):
        forms=platform_forms(name,record,'cljs')
        declarations=[f for f in forms if _top(f,'ns')]
        if len(declarations)!=1:
            return ()
        ns=declarations[0]
        if len(ns.children)<2 or not literal_symbol(ns.children[1]):
            return ()
        seen=set()
        for clause in ns.children[2:]:
            if clause.kind!='list' or not clause.children:
                continue
            head=_sym(clause.children[0])
            if head in seen:
                return ()
            seen.add(head)
            if head not in {':require',':require-macros'}:
                continue
            for spec in clause.children[1:]:
                if _libspec_issue(spec,'cljs'):
                    return ()
                lib=spec.children[0] if spec.kind=='vector' else spec
                if not literal_symbol(lib):
                    return ()
                opts=spec.children[1:] if spec.kind=='vector' else ()
                pairs={_sym(opts[i]):opts[i+1] for i in range(0,len(opts),2)}
                if ':rename' in pairs:
                    referred={_sym(v) for key in (':refer',':refer-macros')
                              for v in (pairs[key].children if key in pairs else ())}
                    if any(_sym(v) not in referred for v in pairs[':rename'].children[::2]):
                        return ()
                for i in range(0,len(opts),2):
                    key=_sym(opts[i]);val=opts[i+1]
                    if key in {':as',':as-alias'} and not literal_symbol(val):
                        return ()
                    if key in {':refer',':refer-macros'} and any(not literal_symbol(v) for v in val.children):
                        return ()
        return declarations

    ns=namespace_forms(path,source)
    if len(ns)!=1:
        return [],[],0
    explicit={r['namespace'] for r in source.get('macro_require_specs',{}).get('requires',[])}
    out=[];unresolved=[];overflow=0
    for clause in ns[0].children[2:]:
        if not _top(clause,':require'):
            continue
        for spec in clause.children[1:]:
            if _libspec_issue(spec,'cljs'):
                continue
            options=spec.children[1:] if spec.kind=='vector' else ()
            pairs={_sym(options[i]):options[i+1] for i in range(0,len(options),2)}
            name=_sym(spec.children[0]) if spec.kind=='vector' else _sym(spec)
            # Explicit sugar controls this spec; alias-only does not load it.
            if name in explicit or any(k in pairs for k in (':as-alias',':include-macros',':refer-macros')):
                continue
            targets=runtime_index.get(name,[])
            if len(targets)>1:
                if len(unresolved)<MAX_UNRESOLVED:
                    unresolved.append({'namespace':name,'line':spec.line,'column':spec.col,
                        'reason':'implicit_macro_runtime_namespace_ambiguous',
                        'candidate_provider_count':len(targets)})
                else:
                    overflow+=1
                continue
            if len(targets)!=1 or targets[0]==path:
                continue
            runtime_path=targets[0];runtime=files[runtime_path]
            if not namespace_forms(runtime_path,runtime):
                continue
            runtime_specs=runtime.get('macro_require_specs',{})
            if runtime_specs.get('overflow_count') or runtime_specs.get('limitations'):
                continue
            self_requires=[r for r in runtime_specs.get('requires',[])
                           if r['namespace']==name and r.get('origin')==':require-macros']
            # Only a literal unambiguous self-require is covered here.
            if len(self_requires)!=1:
                continue
            if len(out)>=MAX_DECLARATIONS:
                overflow+=1;continue
            public=set()
            providers=provider_index.get(name,[])
            if len(providers)==1:
                declarations,truncated=provider_defmacros[providers[0][0]]
                provider_path=providers[0][0]
                replaced={d[0] for f in platform_forms(provider_path,files[provider_path],'clj')
                          if (d:=_definition(f)) and d[1]!='macro'}
                if not truncated:
                    public={n for n,defs in declarations.items()
                            if len(defs)==1 and defs[0]['visibility']=='public' and n not in replaced}
            runtime_forms=platform_forms(runtime_path,runtime,'cljs')
            runtime_names={d[0] for f in runtime_forms
                           if (d:=_definition(f))}
            runtime_refers,refer_limitations=_explicit_refers(list(runtime_forms),'cljs')
            runtime_names.update(runtime_refers)
            if refer_limitations:
                public=set()
            refs=list(pairs[':refer'].children) if ':refer' in pairs else []
            # Rename inference and macro/function collisions need compiler
            # export context. Keep their names unclassified in this subset.
            chosen=[f for f in refs if _sym(f) in public and _sym(f) not in runtime_names
                    and ':rename' not in pairs]
            chosen_names={_sym(f) for f in chosen}
            ns_record=namespaces[runtime_path]['cljs']
            self_req=self_requires[0]
            out.append({'namespace':name,'line':spec.line,'column':spec.col,
                'phase':'compile_time_macro','origin':':require/implicit-self-macro-loading',
                'requested_alias':_sym(pairs[':as']) if ':as' in pairs else None,
                'requested_refer_macros':[_sym(f) for f in chosen],
                'requested_macro_sites':[{'name':_sym(f),'line':f.line,'column':f.col} for f in chosen],
                'implicit_loading_evidence':{
                    'runtime_namespace_evidence':evidence(runtime_path,runtime,
                        ns_record.get('namespace_line'),ns_record.get('namespace_column')),
                    'self_macro_require_evidence':evidence(runtime_path,runtime,
                        self_req['line'],self_req['column']),
                    'unclassified_refer_names':[_sym(f) for f in refs if _sym(f) not in chosen_names],
                    'compiler_verified':False}})
    return out,unresolved,overflow
