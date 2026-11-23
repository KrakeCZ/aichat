"""Offline, source-anchored interprocedural Clojure/CLJS candidate flow.

The only admissible callees are already guarded cross-file, matching-platform
project function declarations with a unique literal arity. For these calls we
index positional argument/formal transfer, formal lexical reads and possible
normal tail-return expressions. Each source expression is represented by an
actual symbol-graph node and evidence-bearing *candidate* edge. No compiler,
project code, macro, classloader, subprocess, network or build is executed.

Candidate edges mean possible source-level transfer, NOT guaranteed runtime
execution, CFG path feasibility, type identity, heap effects or dynamic dispatch.
"""
from __future__ import annotations

from .clojure_scope import _fn_arities
from .clojure_type_hints import literal_type_tag
from .clojure_function_scope import function_contract_plan
from analyzer.symbol_graph import _node_id

MAX_CROSS_FUNCTION_SITES = 2000
MAX_ARGS_PER_SITE = 64
MAX_TAIL_DEPTH = 48
MAX_RETURN_SITES = 48
MAX_LEXICAL_READ_SITES = 128
MAX_FLOW_NODES_PER_FILE = 12000


def _walk(form):
    if form.kind == 'opaque':
        return
    if form.kind == 'list' and form.children:
        head=form.children[0]
        name=head.value if head.kind=='atom' else ''
        if name in {'quote','clojure.core/quote','comment','clojure.core/comment'}:
            return
        yield form
    for child in form.children:
        yield from _walk(child)


def _call_index(forms):
    """Index source locations once; duplicate sites remain unresolvable."""
    positions={}
    for form in forms:
        for nested in _walk(form):
            head=nested.children[0]
            position=(head.line,head.col)
            if position not in positions:
                positions[position]=nested
            else:
                positions[position]=None
    return positions


def _function_arities(forms, target):
    from .clojure_forms import _definition
    candidates=[]
    for form in forms:
        declared=_definition(form)
        if (declared is None or declared[1]!='function' or
                declared[2].line!=target['line'] or
                declared[2].col!=target['column'] or
                declared[0]!=target['name']):
            continue
        parts=list(form.children[2:])
        if parts and parts[0].kind=='string':parts.pop(0)
        if parts and parts[0].kind=='map':parts.pop(0)
        if parts and parts[-1].kind=='map':parts.pop()
        arities=_fn_arities(tuple(parts))
        if arities is not None:
            candidates.extend(arities)
    return candidates if len(candidates) else None


def _source_location(form, sha, path):
    return {'path':path,'line':form.line,'column':form.col,
            'end_line':form.end_line,'end_column':form.end_col,
            'source_sha256':sha}


def _choose_arity(arities, arg_count):
    matches=[]
    for params,_ in arities:
        parts=params.children
        amp=[i for i,p in enumerate(parts) if p.kind=='atom' and p.value=='&']
        if len(amp)>1:
            continue
        if amp:
            if amp[0]!=len(parts)-2 or arg_count<amp[0]:
                continue
        elif len(parts)!=arg_count:
            continue
        matches.append((params,amp[0] if amp else None))
    return matches[0] if len(matches)==1 else None


def _normal_result_sources(expressions):
    """Possible syntactic tail values, never a runtime/path-execution claim.

    Traverse only compiler special forms with documented tail semantics.
    Unknown macros and arbitrary invocations stay a single opaque expression
    candidate; throw/recur contribute no normal result. A tail `if` has a
    synthetic nil path when its else arm is omitted.
    """
    items=[]
    seen=set()
    truncated=False

    def add(form, category):
        nonlocal truncated
        key=(form.line,form.col,form.end_line,form.end_col,category)
        if key in seen:return
        if len(items)>=MAX_RETURN_SITES:
            truncated=True;return
        seen.add(key)
        items.append((form,category))

    def visit(form, depth):
        nonlocal truncated
        if depth>MAX_TAIL_DEPTH or form.kind=='opaque':
            truncated=True;return
        if form.kind!='list' or not form.children or form.children[0].kind!='atom':
            add(form,'literal_or_nonlist_tail_expression');return
        head=form.children[0].value
        parts=form.children
        if head in ('throw','recur'):
            return
        if head in ('do',):
            if len(parts)>1: visit(parts[-1],depth+1)
            else: add(form,'implicit_nil_tail_expression')
            return
        if head=='if':
            if len(parts) not in (3,4):
                truncated=True;return
            visit(parts[2],depth+1)
            if len(parts)==4: visit(parts[3],depth+1)
            else: add(form,'implicit_nil_else_branch')
            return
        if head in ('let*','loop*'):
            if len(parts)<2 or parts[1].kind!='vector' or len(parts[1].children)%2:
                truncated=True;return
            if len(parts)>2: visit(parts[-1],depth+1)
            else: add(form,'implicit_nil_tail_expression')
            return
        if head=='try':
            body=[];catch=[];finally_seen=False
            for part in parts[1:]:
                kind=(part.children[0].value if part.kind=='list' and part.children and
                      part.children[0].kind=='atom' else None)
                if kind=='catch':
                    if finally_seen or len(part.children)<4:
                        truncated=True;return
                    catch.append(part)
                elif kind=='finally':
                    if finally_seen or len(part.children)<2:
                        truncated=True;return
                    finally_seen=True
                elif catch or finally_seen:
                    truncated=True;return
                else:body.append(part)
            if body: visit(body[-1],depth+1)
            else: add(form,'implicit_nil_try_body')
            for part in catch:visit(part.children[-1],depth+1)
            return
        # `let`, `loop` and `fn` are macros or compiler-recognized function
        # syntax depending on the dialect. Never expand them or claim their
        # body value is a proven return. Keep the whole expression as source.
        add(form,'unexpanded_tail_expression')

    if expressions:
        visit(expressions[-1],0)
    else:
        # Empty fn bodies are source-valid in some forms and yield nil.
        return [],False
    return items,truncated


def _return_sources(params, body):
    plan=function_contract_plan(params,body)
    if plan is None:return None,True
    _,ordinary,_=plan
    return _normal_result_sources(ordinary)


def _flow_record(graph, target, arity_index):
    flow=graph.get('clojure_semantic_analysis',{}).get(target.get('qualname',''))
    if not isinstance(flow,dict):return None
    arities=flow.get('arities',[])
    if not isinstance(arities,list) or arity_index>=len(arities):return None
    selected=arities[arity_index]
    if not isinstance(selected,dict):return None
    nodes=selected.get('nodes',[])
    return {'entry':next((n['id'] for n in nodes if n.get('kind')=='entry'),None),
            'normal_return':next((n['id'] for n in nodes if n.get('kind')=='normal_return'),None),
            'lexical_reads':selected.get('reads',[]),
            'unknown':selected.get('unknown',[])}


def _source_node(graph, path, parent, name, kind, form, source_hash, cache):
    """Stable actual Inside File node; a name and location alone are not IDs."""
    key=(path,kind,parent,name,form.line,form.col,form.end_line,form.end_col)
    if key in cache:return cache[key]
    if len(graph['nodes'])>=MAX_FLOW_NODES_PER_FILE:
        return None
    ident=_node_id(path,kind+':'+parent+':'+name,form.line,form.col)
    if any(n['id']==ident for n in graph['nodes']):
        # Never overwrite a preexisting graph node with guessed identity.
        return None
    graph['nodes'].append({
        'id':ident,'path':path,'name':name,'qualname':kind+':'+name,
        'kind':kind,'line':form.line,'column':form.col,
        'end_line':form.end_line,'end_column':form.end_col,
        'parent':parent,'confidence':'source_anchored_candidate',
        'source_sha256':source_hash,'runtime_execution_confirmed':False})
    graph['edges'].append({
        'source':parent,'target':ident,
        'source_path':path,'target_path':path,
        'relation':'clojure_contains_source_expression',
        'resolution':'source_syntax_node_location_only',
        'analyzer':'clojure-source-interprocedural',
        'runtime_execution_confirmed':False,
        'evidence':_source_location(form,source_hash,path)})
    cache[key]=ident
    return ident


def _source_edge(graph, source, target, src_path, dst_path, relation, evidence, **attrs):
    graph['edges'].append({'source':source,'target':target,
         'source_path':src_path,'target_path':dst_path,
         'relation':relation,'resolution':'source_interprocedural_candidate_only',
         'analyzer':'clojure-source-interprocedural',
         'runtime_execution_confirmed':False,
         'compiler_binding_confirmed':False,
         'evidence':evidence,**attrs})


def _same_source_location(read, pattern):
    loc=read.get('binding',{}).get('location',{})
    return (read.get('binding',{}).get('kind')=='parameter' and
      loc.get('line')==pattern.line and loc.get('column')==pattern.col and
      loc.get('end_line')==pattern.end_line and
      loc.get('end_column')==pattern.end_col)


def attach_interprocedural_argument_candidates(graphs, files):
    source_calls = {'clojure_call_candidate', 'clojure_referred_call_candidate',
                    'clojure_local_var_reference_candidate'}
    arity_cache={}
    anchor_cache={}
    collected_calls=[]
    for source_path,graph in sorted(graphs.items()):
        emitted=[]
        missing=0
        call_index=None
        for edge in graph['edges']:
            if edge.get('relation') not in source_calls:
                continue
            # A compiler-backed runtime target is NOT implied by a candidate.
            if edge.get('analysis_phase')=='macro_expansion':
                continue
            # A shared .cljc source has no single selected compiler target.
            # Do not assign parameter/argument identities across variants.
            source_platform=edge.get('source_platform')
            target_path=edge.get('target_path')
            target_platform=edge.get('target_platform',
                source_platform if target_path==source_path else None)
            if (source_platform not in {'clj','cljs'} or
                    source_platform!=target_platform):
                continue
            target_data=files.get(target_path)
            source_data=files.get(source_path)
            if not (target_data and source_data):
                continue
            target=next((node for node in graphs[target_path]['nodes']
                if node['id']==edge['target'] and node['kind']=='function'),None)
            if target is None:
                continue
            evidence=edge.get('evidence',{})
            if len(emitted)>=MAX_CROSS_FUNCTION_SITES:
                missing+=1
                continue
            if call_index is None:
                call_index=_call_index(source_data['forms'])
            call=call_index.get((evidence.get('line'),evidence.get('column')))
            if call is None or not call.children or len(call.children)-1>MAX_ARGS_PER_SITE:
                continue
            target_key=(target_path,target['id'])
            if target_key not in arity_cache:
                arity_cache[target_key]=_function_arities(target_data['forms'],target)
            arities=arity_cache[target_key]
            if arities is None:
                continue
            selection=_choose_arity(arities,len(call.children)-1)
            if selection is None:
                continue
            params,rest_at=selection
            arity_index=next((i for i,(p,_body) in enumerate(arities)
                         if p is params),None)
            if arity_index is None:continue
            target_body=arities[arity_index][1]
            return_sources,return_truncated=_return_sources(params,target_body)
            flow=_flow_record(graphs[target_path],target,arity_index)
            bindings=[]
            args=call.children[1:]
            limit=rest_at if rest_at is not None else len(params.children)
            for index in range(limit):
                parameter=params.children[index]
                binding={'position':index,
                    'argument_expression':_source_location(args[index],
                        source_data['sha'],source_path),
                    'parameter_pattern':_source_location(parameter,
                        target_data['sha'],target_path),
                    'parameter_kind':parameter.kind,
                    'provenance':'literal_position_correspondence_not_runtime_dataflow'}
                if (tag:=literal_type_tag(parameter)) is not None:
                    binding['declared_parameter_type_hint']=tag
                bindings.append(binding)
            if rest_at is not None:
                param=params.children[rest_at+1]
                bindings.append({'position':rest_at,'variadic_rest_sequence_candidate':True,
                    'argument_expressions':[_source_location(arg,source_data['sha'],
                        source_path) for arg in args[rest_at:]],
                    'parameter_pattern':_source_location(param,target_data['sha'],target_path),
                    'parameter_kind':param.kind,
                    'provenance':'literal_rest_parameter_correspondence_not_runtime_dataflow'})
            item={'caller_symbol':edge['source'],'target_declaration':edge['target'],
                  'source_path':source_path,'target_path':target_path,
                  'call_head':evidence,'parameter_list':_source_location(params,
                      target_data['sha'],target_path),
                  'argument_mapping_candidates':bindings,
                  'proof':'unique_guarded_project_source_call_candidate_and_matching_literal_arity',
                  'runtime_call_target_confirmed':False,
                  'dataflow_execution_confirmed':False,
                  'return_flow_confirmed':False,
                  'target_platform':target_platform,
                  'source_platform':edge.get('source_platform')}
            # Source-only interprocedural bridge nodes use the SAME Inside
            # File graph store. Separate candidate relations never replace
            # the original guarded call or assert a runtime edge.
            callee_graph=graphs[target_path]
            call_result=_source_node(graph,source_path,edge['source'],
                'call-result:'+str(call.line)+':'+str(call.col),
                'clojure_call_result_candidate',call,source_data['sha'],anchor_cache)
            site_summary={'arity_index':arity_index,'cfg_entry_node':None,
                'cfg_normal_return_node':None,'possible_normal_results':[],
                'normal_result_sites_truncated':return_truncated,
                'parameter_read_count':0,'parameter_reads_truncated':False,
                'call_result_node':call_result,'candidate_graph_bridges':0}
            if flow:
                site_summary['cfg_entry_node']=flow['entry']
                site_summary['cfg_normal_return_node']=flow['normal_return']
                site_summary['callee_cfg_unknown_count']=len(flow['unknown'])
            # Call-site -> existing function declaration, with whole-call
            # source anchoring (the pre-existing edge proves only call HEAD).
            if call_result:
                _source_edge(graph,call_result,target['id'],source_path,target_path,
                    'clojure_call_site_to_declaration_candidate',
                    _source_location(call,source_data['sha'],source_path),
                    arity_index=arity_index)
                site_summary['candidate_graph_bridges']+=1
            for binding in bindings:
                position=binding['position']
                arg_forms=(args[position:]
                    if binding.get('variadic_rest_sequence_candidate')
                    else args[position:position+1])
                pattern=params.children[position+1] if rest_at is not None and position==rest_at else params.children[position]
                formal=_source_node(callee_graph,target_path,target['id'],
                    'formal:'+str(arity_index)+':'+str(position),
                    'clojure_formal_pattern_candidate',pattern,target_data['sha'],anchor_cache)
                binding['formal_graph_node']=formal
                binding['actual_graph_nodes']=[]
                if formal:
                    observations=[]
                    if flow:
                        observations=[r for r in flow['lexical_reads']
                                      if _same_source_location(r,pattern)]
                    binding['lexical_read_candidates']=[
                        {'source':r['location'],'name':r['symbol'],
                         'evaluation_phase':r.get('evaluation_phase','immediate'),
                         'binding_id':r['binding']['id']}
                        for r in observations[:MAX_LEXICAL_READ_SITES]]
                    binding['lexical_reads_truncated']=(
                        len(observations)>MAX_LEXICAL_READ_SITES)
                    site_summary['parameter_read_count']+=len(binding['lexical_read_candidates'])
                    site_summary['parameter_reads_truncated'] |=binding['lexical_reads_truncated']
                for actual_position,arg_form in enumerate(arg_forms, start=position):
                    actual=_source_node(graph,source_path,edge['source'],
                        'actual:'+str(call.line)+':'+str(call.col)+':'+str(actual_position),
                        'clojure_actual_argument_candidate',arg_form,source_data['sha'],anchor_cache)
                    if not actual or not formal:continue
                    binding['actual_graph_nodes'].append(actual)
                    _source_edge(graph,actual,formal,source_path,target_path,
                        'clojure_argument_to_formal_candidate',
                        _source_location(arg_form,source_data['sha'],source_path),
                        formal_pattern_evidence=_source_location(pattern,target_data['sha'],target_path),
                        arity_index=arity_index,argument_position=actual_position,
                        variadic_rest=bool(binding.get('variadic_rest_sequence_candidate')))
                    site_summary['candidate_graph_bridges']+=1
            if return_sources is not None:
                for candidate,category in return_sources:
                    return_node=_source_node(callee_graph,target_path,target['id'],
                        'tail:'+str(arity_index)+':'+str(candidate.line)+':'+str(candidate.col)+':'+category,
                        'clojure_possible_return_expression',candidate,target_data['sha'],anchor_cache)
                    location=_source_location(candidate,target_data['sha'],target_path)
                    site_summary['possible_normal_results'].append({
                        'expression':location,'category':category,
                        'source_graph_node':return_node,'runtime_result_confirmed':False})
                    if return_node and call_result:
                        _source_edge(callee_graph,return_node,call_result,target_path,source_path,
                            'clojure_possible_return_to_call_site_candidate',location,
                            destination_call_site=_source_location(call,source_data['sha'],source_path),
                            arity_index=arity_index,source_platform=target_platform)
                        site_summary['candidate_graph_bridges']+=1
            item['interprocedural_control_dataflow_candidates']=site_summary
            emitted.append(item)
            collected_calls.append((edge['source'],target['id'],source_path,target_path))
        graph['clojure_interprocedural_argument_candidates']={
            'status':'source_candidate_only_no_interprocedural_cfg_certification',
            'calls':emitted,'limit':MAX_CROSS_FUNCTION_SITES,
            'omitted_due_to_limit':missing}

    # Source-candidate SCCs: useful for architectural cycles without claiming
    # real recursion or dynamic reachability. Tarjan runs on the bounded set of
    # already-guarded and arity-matched function declarations only.
    adjacency={}
    for caller,callee,src,dst in collected_calls:
        adjacency.setdefault(caller,set()).add(callee)
        adjacency.setdefault(callee,set())
    discovery={};low={};stack=[];on_stack=set();components=[];serial=0
    def strong(v):
        nonlocal serial
        discovery[v]=serial;low[v]=serial;serial+=1
        stack.append(v);on_stack.add(v)
        for w in sorted(adjacency[v]):
            if w not in discovery:
                strong(w);low[v]=min(low[v],low[w])
            elif w in on_stack:
                low[v]=min(low[v],discovery[w])
        if low[v]==discovery[v]:
            members=[]
            while True:
                w=stack.pop();on_stack.remove(w);members.append(w)
                if w==v:break
            if len(members)>1 or v in adjacency[v]:
                components.append(tuple(sorted(members)))
    # Fail closed instead of hitting Python recursion limits in huge projects.
    if len(adjacency)<=500:
        for symbol in sorted(adjacency):
            if symbol not in discovery:strong(symbol)
        report={'status':'bounded_source_candidate_call_cycles',
                'components':[{'member_symbols':list(comp),
                               'candidate_only':True} for comp in sorted(components)],
                'observed_execution':False,'edge_count':len(collected_calls)}
    else:
        report={'status':'unavailable_candidate_graph_exceeds_500_functions',
                'components':[],'observed_execution':False,
                'edge_count':len(collected_calls)}
    for graph in graphs.values():
        graph['clojure_interprocedural_source_cycle_candidates']=report
