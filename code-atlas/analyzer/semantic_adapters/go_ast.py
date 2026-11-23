"""Go source adapter via the official go/parser standard-library AST.

Parses without running target code, invoking go list, or contacting registries.
Syntactic declarations are verified. A local go/types pass resolves selected
direct calls, including cross-file/cross-package links, only for complete
packages with resolved types. Unresolved names remain candidates.
Build tags and runtime interface dispatch are never guessed.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from analyzer.symbol_graph import _node_id, _file_id

BRIDGE = Path(__file__).with_name('AtlasGoAstBridge.go')
MAX_SOURCE = 1_000_000
MAX_STDOUT = 30_000_000

def bridge_executable_name(platform=None):
    """Give Go-built helper a Windows executable suffix when required."""
    return 'atlas-go-ast.exe' if (platform or os.name) == 'nt' else 'atlas-go-ast'


def _module_name(root):
    manifest=root/'go.mod'
    try:
        # Only the standard `module` directive, no evaluation.
        for line in manifest.read_text('utf8').splitlines():
            line=line.split('//',1)[0].strip()
            if line.startswith('module '):
                return line.split(None,1)[1].strip().strip('"')
    except OSError: pass
    return ''


def analyze_go(root:Path,candidates:dict,_graph:dict)->tuple[dict,list]:
    paths=sorted(p for p,v in candidates.items() if v.get('language')=='go')
    if not paths:return {},[]
    go=shutil.which('go')
    if not go:return {},[{'reason':'go_parser_unavailable','detail':'Go toolchain is not installed'}]
    root=root.resolve(); files={}; warnings=[]
    for p in paths:
        f=(root/p).resolve()
        if not f.is_relative_to(root) or not f.is_file() or f.stat().st_size>MAX_SOURCE:
            warnings.append({'path':p,'reason':'go_source_skipped'});continue
        raw=f.read_bytes()
        files[p]={'raw':raw,'lines':raw.decode('utf8',errors='replace').splitlines(),
                  'sha':hashlib.sha256(raw).hexdigest()}
    if not files:return {},warnings
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-go-ast-') as d:
            t=Path(d);listing=t/'manifest.json';listing.write_text(json.dumps({'root':str(root),'files':list(files),'module':_module_name(root)}),'utf8')
            binary=t/bridge_executable_name();env={**os.environ,'GO111MODULE':'off','GOPROXY':'off','GOSUMDB':'off','GOTOOLCHAIN':'local','GOWORK':'off'}
            p=subprocess.run([go,'build','-o',str(binary),str(BRIDGE),str(BRIDGE.with_name('AtlasGoTypesBridge.go'))],capture_output=True,text=True,
                             cwd=t,env=env,timeout=85)
            if p.returncode:raise RuntimeError(p.stderr[:400])
            p=subprocess.run([str(binary),str(listing)],capture_output=True,cwd=t,timeout=30)
            if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[:400])
            if len(p.stdout)>MAX_STDOUT:raise ValueError('Go AST output limit exceeded')
            records=json.loads(p.stdout)
    except (OSError,subprocess.TimeoutExpired,RuntimeError,ValueError) as e:
        return {},warnings+[{'reason':'go_ast_failed','details':str(e)[:260]}]
    reported = {item.get('path') for item in records if isinstance(item,dict)}
    for path in files:
        if path not in reported:
            warnings.append({'reason':'go_ast_missing_source_record','path':path})
    module=_module_name(root)
    graphs={};decls=defaultdict(list);packages={};by_dir=defaultdict(list)
    for rec in records:
        path=rec.get('path')
        if path not in files:continue
        if rec.get('parse_error'):
            warnings.append({'path':path,'reason':'go_parse_error','details':rec['parse_error'][:250]});continue
        raw=files[path]; package=rec.get('package','')
        packages[path]=package;by_dir[str(Path(path).parent)].append(path)
        file_node={'id':_file_id(path),'path':path,'kind':'file','name':Path(path).name,
                   'line':1,'confidence':'verified_file'}
        graphs[path]={'nodes':[file_node], 'edges':[],'parser':'go_stdlib_ast_typed' if rec.get('type_checked') else 'go_stdlib_ast','hash':raw['sha'],
              'caution':'Go AST + optional go/types: direct calls are resolved only in fully type-checked local packages. Other links remain candidates; build constraints, cgo and runtime dispatch are not proven.'}
        for n in rec.get('nodes',[]):
            line=int(n.get('line',0));col=int(n.get('column',0))
            if line<1 or line>len(raw['lines']):continue
            qual=n['qual']; node={'id':_node_id(path,qual,line,col),'path':path,'kind':n['kind'],
                    'name':n['name'],'qualname':qual,'line':line,'column':col,
                    'end_line':int(n.get('end_line',line)),'confidence':'verified_syntax','_parent':n.get('parent','')}
            graphs[path]['nodes'].append(node)
            decls[(path,qual)].append(node)
    def evidence(path,line):
        f=files[path];lines=f['lines']
        return {'path':path,'line':line,'source_sha256':f['sha'],
                'snippet':lines[line-1].strip()[:200] if 0<line<=len(lines) else ''}
    for path,graph in graphs.items():
        for n in graph['nodes'][1:]:
            parent=n.pop('_parent','')
            pars=decls.get((path,parent),[]) if parent else [graph['nodes'][0]]
            if len(pars)!=1:pars=[graph['nodes'][0]]
            n['parent']=pars[0]['id']
            graph['edges'].append({'source':pars[0]['id'],'target':n['id'],
               'source_path':path,'target_path':path,'relation':'declares',
               'resolution':'verified_syntax','analyzer':'go-stdlib-ast','evidence':evidence(path,n['line'])})
    # A package is a group of files in one directory; module path from go.mod
    # makes qualified local imports unambiguous without any project-specific names.
    by_package=defaultdict(list)
    for path in graphs:by_package[(str(Path(path).parent),packages[path])].append(path)
    for rec in records:
        path=rec['path'];graph=graphs.get(path)
        if not graph:continue
        imps={}
        for spec in rec.get('imports',[]):
            pkg=spec['path'];alias=spec.get('alias') or pkg.rsplit('/',1)[-1]
            if alias in ('.','_') or not module or not pkg.startswith(module+'/'):continue
            sub=pkg[len(module)+1:]
            if not sub or sub.startswith('/') or '..' in Path(sub).parts:continue
            possible=by_package.get((sub,sub.rsplit('/',1)[-1]),[])
            # The Go package declaration may differ from its directory name:
            # collect by actual local directory rather than assume package name.
            if not possible:possible=[p for p in by_dir.get(sub,[]) if p in graphs]
            if possible:imps[alias]=possible
        # go/types identifies the actual declaration via types.Info.Uses;
        # never promote an AST name match to a verified call.
        typed_sites=set()
        for call in rec.get('typed_calls',[]):
            typed_sites.add((call.get('owner'),int(call.get('line',0)),int(call.get('column',0))))
            owner=decls.get((path,call.get('owner','')),[])
            destination=decls.get((call.get('target_path',''),call.get('target_qual','')),[])
            destination=[n for n in destination if n['line']==int(call.get('target_line',0))]
            if len(owner)!=1 or len(destination)!=1:continue
            graph['edges'].append({'source':owner[0]['id'],'target':destination[0]['id'],
                'source_path':path,'target_path':destination[0]['path'],
                'relation':'calls','resolution':'resolved_static','analyzer':'go-types',
                'evidence':evidence(path,int(call['line']))})
        for call in rec.get('calls',[]):
            if (call.get('owner'),int(call.get('line',0)),int(call.get('column',0))) in typed_sites:continue
            if call.get('shadow'):continue
            name=call['name'];recv=call.get('receiver','');owner=call.get('owner','')
            owners=decls.get((path,owner),[])
            if len(owners)!=1:continue
            if recv:
                targets=imps.get(recv,[])
            else:
                targets=by_dir.get(str(Path(path).parent),[])
            definitions=[n for target in targets for n in graphs[target]['nodes'][1:]
                         if n['kind']=='function' and n['name']==name]
            if len(definitions)!=1:continue
            target=definitions[0]
            # Go's namespace checks will be added with go/types; never call this
            # 'resolved_static' based only on an AST name.
            graph['edges'].append({'source':owners[0]['id'],'target':target['id'],
                'source_path':path,'target_path':target['path'],'relation':'call_candidate',
                'resolution':'go_ast_name_candidate','analyzer':'go-stdlib-ast',
                'evidence':evidence(path,int(call['line']))})
    return graphs,warnings
