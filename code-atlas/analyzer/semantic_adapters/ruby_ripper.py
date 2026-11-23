"""Ruby Ripper AST adapter, without requiring or running repository code.

Ripper verifies declarations syntactically. Ruby dispatch and autoloading are
highly dynamic; any name-matched method edge is a candidate, never proven.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from analyzer.symbol_graph import _node_id, _file_id

BRIDGE=Path(__file__).with_name('AtlasRubyRipperBridge.rb')
MAX_FILES=300
MAX_SOURCE=1_000_000

def analyze_ruby(root:Path,candidates:dict,_graph:dict):
    paths=sorted(p for p,v in candidates.items() if v.get('language')=='ruby')
    if not paths:return {},[]
    ruby=shutil.which('ruby')
    if not ruby:return {},[{'reason':'ruby_ripper_unavailable'}]
    root=root.resolve();warnings=[];files={}
    for path in paths:
        p=(root/path).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size>MAX_SOURCE:
            warnings.append({'reason':'ruby_source_skipped','path':path});continue
        raw=p.read_bytes();files[path]={'sha':hashlib.sha256(raw).hexdigest(),'lines':raw.decode('utf-8',errors='replace').splitlines()}
    if not files:return {},warnings
    # Parse in bounded batches, then resolve require_relative edges across the
    # complete project. The previous MAX_FILES slice silently omitted the rest.
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-ruby-ast-') as d:
            manifest=Path(d)/'files.json'
            selected=list(files)
            for offset in range(0,len(selected),MAX_FILES):
                manifest.write_text(json.dumps({'root':str(root),
                                                'files':selected[offset:offset + MAX_FILES]}),'utf8')
                proc=subprocess.run([ruby,'--disable-gems',str(BRIDGE),str(manifest)],
                                    cwd=d,capture_output=True,timeout=30)
                if proc.returncode:
                    raise RuntimeError(f'Ruby batch {offset // MAX_FILES + 1}: '
                                       + proc.stderr.decode(errors='replace')[:240])
                if len(proc.stdout)>30_000_000:
                    raise RuntimeError('Ruby AST batch output limit exceeded')
                group=json.loads(proc.stdout)
                if not isinstance(group,list):
                    raise ValueError('Ruby AST bridge produced a non-list response')
                records.extend(group)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as e:
        return {},warnings+[{'reason':'ruby_ripper_failed','details':str(e)[:240]}]
    reported = {item.get('path') for item in records if isinstance(item,dict)}
    for path in files:
        if path not in reported:
            warnings.append({'reason':'ruby_ripper_missing_source_record','path':path})
    graphs={};decls=defaultdict(list)
    def evidence(path,line):
        info=files[path]
        return {'path':path,'line':line,'source_sha256':info['sha'],
                'snippet':info['lines'][line-1].strip()[:200] if 0<line<=len(info['lines']) else ''}
    for item in records:
        path=item.get('path')
        if path not in files:continue
        if item.get('parse_error'):
            warnings.append({'reason':'ruby_parse_error','path':path,'details':item['parse_error']});continue
        file_node={'id':_file_id(path),'path':path,'kind':'file','name':Path(path).name,'line':1,'confidence':'verified_file'}
        graphs[path]={'nodes':[file_node],'edges':[], 'hash':files[path]['sha'],'parser':'ruby_ripper_ast',
                      'caution':'Ruby Ripper AST verifies declarations, not dynamic dispatch, mixins or autoload resolution.'}
        for nd in item.get('nodes',[]):
            ln=int(nd.get('line',0));col=int(nd.get('column',0))
            if not 0<ln<=len(files[path]['lines']):continue
            q=nd['qual'];node={'id':_node_id(path,q,ln,col),'name':nd['name'],'qualname':q,
                'kind':nd['kind'],'path':path,'line':ln,'column':col,'confidence':'verified_syntax',
                '_parent':nd.get('parent','')}
            graphs[path]['nodes'].append(node);decls[(path,q)].append(node)
    for path,g in graphs.items():
        for node in g['nodes'][1:]:
            parent=node.pop('_parent','')
            opts=decls.get((path,parent),[]) if parent else [g['nodes'][0]]
            if len(opts)!=1:opts=[g['nodes'][0]]
            node['parent']=opts[0]['id']
            g['edges'].append({'source':opts[0]['id'],'target':node['id'],
                'source_path':path,'target_path':path,'relation':'declares','resolution':'verified_syntax',
                'analyzer':'ruby-ripper','evidence':evidence(path,node['line'])})
    for item in records:
        path=item.get('path');g=graphs.get(path)
        if not g:continue
        required=[]
        for r in item.get('requires',[]):
            value=r['value']; candidate=(Path(path).parent/value)
            if candidate.suffix!='.rb':candidate=Path(str(candidate)+'.rb')
            # Only repo-relative paths, never absolute paths or traversal.
            dest=(root/candidate).resolve()
            if dest.is_relative_to(root):
                target=dest.relative_to(root).as_posix()
                if target in graphs:required.append(target)
        for call in item.get('calls',[]):
            if call.get('receiver') or call['name']=='require_relative':continue
            owners=decls.get((path,call.get('owner','')),[]) if call.get('owner') else [g['nodes'][0]]
            if len(owners)!=1:continue
            # Same-file lexical name alone is insufficient for Ruby dynamic dispatch.
            # A required file provides a plausible method origin, never a proof.
            targets=[n for dest in required for n in graphs[dest]['nodes'][1:]
                     if n['kind']=='function' and n['name']==call['name']]
            if len(targets)!=1:continue
            target=targets[0];g['edges'].append({'source':owners[0]['id'],'target':target['id'],
                 'source_path':path,'target_path':target['path'],'relation':'call_candidate',
                 'resolution':'ruby_require_candidate','analyzer':'ruby-ripper',
                 'evidence':evidence(path,int(call['line']))})
    return graphs,warnings
