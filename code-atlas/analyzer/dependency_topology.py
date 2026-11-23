"""Observed graph topology. No guessed software patterns or project-specific names.

The primary graph is explicitly split by evidence class:
- confirmed: exact syntax + resolved static reference
- candidates: bounded possible runtime paths; useful but not proof of execution

Connected components/cycles refer ONLY to recorded directed file dependencies.
Neither graph connectedness nor import cycles proves a runtime architecture.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any

CONFIRMED = frozenset({"exact_syntax", "resolved_static"})
CANDIDATE = frozenset({"candidate", "unresolved"})


def _components(files: list[str], pairs: set[tuple[str,str]]) -> list[list[str]]:
    links: dict[str,set[str]] = {f:set() for f in files}
    for src,tgt in pairs:
        links[src].add(tgt)
        links[tgt].add(src)
    unseen=set(files)
    regions=[]
    while unseen:
        start=min(unseen); unseen.remove(start); todo=[start]; group=[]
        while todo:
            current=todo.pop(); group.append(current)
            for neighbour in sorted(links[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour); todo.append(neighbour)
        regions.append(sorted(group))
    return sorted(regions, key=lambda c:(-len(c),c[0]))


def _strong_cycles(files: list[str], pairs: set[tuple[str,str]]) -> list[list[str]]:
    # Iterative Kosaraju; avoids recursion depth on large repositories.
    adj={f:[] for f in files}; reverse={f:[] for f in files}
    for a,b in sorted(pairs):
        adj[a].append(b);reverse[b].append(a)
    seen=set();order=[]
    for start in files:
        if start in seen: continue
        seen.add(start); stack=[(start, iter(adj[start]))]
        while stack:
            node,neighbors=stack[-1]
            nxt=next(neighbors,None)
            if nxt is None:
                order.append(node);stack.pop()
            elif nxt not in seen:
                seen.add(nxt);stack.append((nxt,iter(adj[nxt])))
    seen.clear();cycles=[]
    for start in reversed(order):
        if start in seen:continue
        seen.add(start);todo=[start];group=[]
        while todo:
            node=todo.pop();group.append(node)
            for neighbour in reverse[node]:
                if neighbour not in seen:
                    seen.add(neighbour);todo.append(neighbour)
        if len(group)>1:cycles.append(sorted(group))
    return sorted(cycles,key=lambda c:(-len(c),c[0]))


def derive_dependency_topology(graph_data: dict[str,Any]) -> dict[str,Any]:
    """Deterministic analysis over file-to-file edges with existing evidence only."""
    files=sorted(n['id'] for n in graph_data['nodes'] if n.get('type')=='file')
    file_set=set(files)
    reliable=set(); potential=set(); confidence_counts=Counter(); type_counts=Counter()
    evidence_by_pair=defaultdict(list)
    for edge in graph_data.get('edges',[]):
        source,target=edge.get('source'),edge.get('target')
        if source not in file_set or target not in file_set or source==target:continue
        confidence=edge.get('confidence_v3', 'candidate')
        confidence_counts[confidence]+=1
        type_counts[edge.get('type','unknown')]+=1
        pair=(source,target)
        if confidence in CONFIRMED: reliable.add(pair)
        else:potential.add(pair)
        evidence_by_pair[pair].append(edge)
    reliable_components=_components(files,reliable)
    possible_components=_components(files,reliable|potential)
    cycles=_strong_cycles(files,reliable)
    in_degree=Counter(dst for _,dst in reliable)
    out_degree=Counter(src for src,_ in reliable)
    possible_degree=Counter(a for a,_ in potential)
    entry=[f for f in files if out_degree[f]>0 and in_degree[f]==0]
    providers=[f for f in files if in_degree[f]>0 and out_degree[f]==0]
    isolated=[f for f in files if in_degree[f]+out_degree[f]==0]
    cross_language=[]
    language={n['id']:n.get('properties',{}).get('language','unknown') for n in graph_data['nodes'] if n.get('type')=='file'}
    for src,dst in sorted(reliable|potential):
        if language[src]==language[dst]:continue
        evidence=evidence_by_pair[(src,dst)]
        cross_language.append({'source':src,'target':dst,'source_language':language[src],
          'target_language':language[dst], 'confidence': 'confirmed' if (src,dst) in reliable else 'candidate',
          'types':sorted({e.get('type','unknown') for e in evidence}),
          'evidence':[{'path':ev.get('path'),'line':ev.get('line')} for e in evidence for ev in e.get('evidence',[])][:10]})
    return {'schema':'atlas-dependency-topology-1','source_commit':graph_data.get('source_commit',''),
      'basis':'observed_file_edges_only','limitation':'A missing edge never proves independence. Static topology cannot infer runtime execution.',
      'files':len(files), 'confirmed_pairs':len(reliable), 'candidate_pairs':len(potential-reliable),
      'confidence_distribution':dict(sorted(confidence_counts.items())),
      'edge_types':dict(sorted(type_counts.items())),
      'confirmed_regions':[{'files':c,'size':len(c)} for c in reliable_components],
      'including_candidates_regions':[{'files':c,'size':len(c)} for c in possible_components],
      'confirmed_cycles':[{'files':c,'size':len(c)} for c in cycles],
      'entry_candidates':entry,'provider_candidates':providers,'without_confirmed_edges':isolated,
      'cross_language_edges':cross_language,
      'most_connected':[{'path':f,'confirmed_in':in_degree[f],'confirmed_out':out_degree[f],
                          'candidate_out':possible_degree[f]} for f in
                         sorted(files,key=lambda x:(-(in_degree[x]+out_degree[x]),x))[:25]]}
