"""Offline, platform-partitioned Clojure namespace *candidate* evidence.

No classpath, compiler or reader is invoked. The only targets considered are
files actually indexed in the supplied repository. An exact namespace declaration
must be uniquely identifiable in the requested platform; two matching files
are ambiguous even when extensions differ because real classpath is unknown.

Intentionally produces metadata, NOT verified graph edges or runtime facts.
"""
from __future__ import annotations

from pathlib import Path

MAX_REQUIRE_EVIDENCE = 250
MAX_UNRESOLVED_EVIDENCE = 80


def _platforms(path: str) -> tuple[str, ...]:
    suffix=Path(path).suffix.lower()
    return {'.clj': ('clj',), '.cljs': ('cljs',), '.cljc': ('clj', 'cljs')}.get(suffix, ())


def _view(path: str, source: dict, platform: str) -> dict:
    """Only accept Stage40 candidate projections or native file ns form subset."""
    if Path(path).suffix.lower()=='.cljc':
        projected=source.get('platform_projections',{}).get(platform,{})
        if projected.get('status')!='candidate_not_semantically_verified':
            return {'status':'unavailable','reason':projected.get('reason','projection_unavailable')}
        name=projected.get('namespace')
        if not name or projected.get('namespace_issue'):
            return {'status':'unavailable','reason':projected.get('namespace_issue') or 'no_namespace'}
        return {'status':'candidate','namespace':name,
                'namespace_line':projected.get('namespace_line'),
                'namespace_column':projected.get('namespace_column'),
                'requires':projected.get('candidate_requires',[]),
                'sha':source['sha']}
    ns, _aliases, imports, issue=source['namespace']
    if not ns or issue:
        return {'status':'unavailable','reason':issue or 'no_namespace'}
    return {'status':'candidate','namespace':ns[0],
            'namespace_line':ns[1], 'namespace_column':ns[2],
            'requires':[{'namespace':n,'line':line,'column':col} for n,line,col in imports],
            'sha':source['sha']}


def build_platform_namespace_evidence(files: dict[str,dict], source_contexts: dict | None = None) -> dict[str,dict]:
    """Return per-file evidence scoped to clj / cljs, never shared/merged.

    No filesystem operations are permitted here. Target evidence comes from
    actual parsed declarations in ``files``. Malformed or unavailable platforms
    cannot be targets. Missing external namespaces are reported, not fabricated.
    """
    from .clojure_source_context import select_namespace_paths, source_path_evidence
    source_contexts=source_contexts or {}
    views={}
    namespaces:dict[tuple[str,str],list[tuple[str,dict]]]={}
    for path in sorted(files):
        views[path]={}
        for platform in _platforms(path):
            view=_view(path,files[path],platform)
            views[path][platform]=view
            if view['status']=='candidate':
                namespaces.setdefault((platform,view['namespace']),[]).append((path,view))
    output={}
    for path in sorted(files):
        output[path]={}
        for platform,view in views[path].items():
            if view['status']!='candidate':
                output[path][platform]={'status':'unavailable','reason':view['reason'],
                                         'namespace_load_certified':False,
                                         'classpath_resolved':False,
                                         'candidates':[],'unresolved':[]}
                continue
            candidates=[];unresolved=[];overflow=0
            for required in view['requires']:
                name=required.get('namespace')
                if not name or not isinstance(name,str):
                    continue
                targets=namespaces.get((platform,name),[])
                context=source_contexts.get(path)
                selected,context_reason=select_namespace_paths(context,name,platform,[p for p,_ in targets])
                targets=[(p,v) for p,v in targets if p in set(selected)]
                reason=context_reason
                if not reason and not targets: reason='namespace_external_or_not_indexed_in_platform'
                elif not reason and len(targets)>1: reason='ambiguous_namespace_multiple_source_files'
                elif not reason and targets[0][0]==path:reason='self_require_not_cross_file'
                if reason:
                    if len(unresolved)<MAX_UNRESOLVED_EVIDENCE:
                        unresolved.append({'namespace':name,'reason':reason,
                                           'line':required.get('line'),
                                           'column':required.get('column'),
                                           'candidate_target_count':len(targets)})
                    else:overflow+=1
                    continue
                target_path,target=targets[0]
                if len(candidates)>=MAX_REQUIRE_EVIDENCE:
                    overflow+=1
                    continue
                candidates.append({
                    'relation':'clojure_namespace_require_candidate',
                    'resolution':('declared_project_source_paths_candidate_no_effective_basis' if context and context.get('status')=='project_paths_declared' else 'unique_indexed_namespace_only_no_classpath_proof'),
                    'namespace':name,'source_platform':platform,'target_platform':platform,
                    'source_path':path,'target_path':target_path,
                    'source_evidence':{'path':path,'line':required.get('line'),
                                       'column':required.get('column'),
                                       'source_sha256':view['sha']},
                    'target_evidence':{'path':target_path,'line':target['namespace_line'],
                                       'column':target['namespace_column'],
                                       'source_sha256':target['sha']},
                    'requires_build_classpath_confirmation':True,
                    'semantic_resolution_certified':False,
                })
                proof=source_path_evidence(context,name,platform,target_path)
                if proof:candidates[-1]['project_source_path_evidence']=proof
            output[path][platform]={
                'status':'candidate_not_semantically_verified',
                'namespace':view['namespace'],
                # Own ns declaration must be traceable, not just refer edges.
                # Stage43 macro evidence expected these but they were absent.
                'namespace_line':view['namespace_line'],
                'namespace_column':view['namespace_column'],
                'namespace_load_certified':False,
                'classpath_resolved':False,
                'candidate_count':len(candidates),
                'unresolved_count':len(unresolved),
                'overflow_count':overflow,
                'candidates':candidates,
                'unresolved':unresolved,
            }
    return output
