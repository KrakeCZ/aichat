#!/usr/bin/env python3
"""Opt-in JVM/CLJS resource-order reference on embedded synthetic files only.

No project path, source input, build, dependency manager or downloader is accepted.
Explicit -cp ordering is checked; this does NOT claim a tools.deps effective basis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import unquote
from analyzer.semantic_adapters.clojure_forms import analyze_clojure
from atlas_clojure_reference import resolve_jar_dir, preflight_jars
from atlas_cljs_compiler_reference import _jar_preflight

# Expected expanded roots are hand-specified, not calculated by the resolver.
CASES=(
 {'id':'native_after_shared','ns':'acme.lib','paths':['first','second'],
  'sources':{'first/acme/lib.cljc':'shared','second/acme/lib.clj':'native'},'want':'second/acme/lib.clj'},
 {'id':'first_root','ns':'acme.lib','paths':['first','second'],
  'sources':{'first/acme/lib.clj':'first','second/acme/lib.clj':'second'},'want':'first/acme/lib.clj'},
 {'id':'reversed_roots','ns':'acme.lib','paths':['second','first'],
  'sources':{'first/acme/lib.clj':'first','second/acme/lib.clj':'second'},'want':'second/acme/lib.clj'},
 {'id':'shared_fallback','ns':'acme.lib','paths':['first','second'],
  'sources':{'first/acme/lib.cljc':'shared'},'want':'first/acme/lib.cljc'},
 {'id':'hyphen_resource','ns':'acme.my-lib','paths':['first'],
  'sources':{'first/acme/my_lib.clj':'hyphen'},'want':'first/acme/my_lib.clj'},
 {'id':'misdeclared_shadow','ns':'acme.lib','paths':['first','second'],
  'sources':{'first/acme/lib.clj':'wrong_namespace','second/acme/lib.clj':'second'},'want':None},
)
CLJS_CASES=(
 {'id':'cljs_native_after_shared','ns':'acme.lib','paths':['first','second'],
  'sources':{'first/acme/lib.cljc':'shared','second/acme/lib.cljs':'native'},'want':'second/acme/lib.cljs'},
 {'id':'cljs_root_order','ns':'acme.lib','paths':['second','first'],
  'sources':{'first/acme/lib.cljs':'first','second/acme/lib.cljs':'second'},'want':'second/acme/lib.cljs'},
)


def prepare(root, case, platform):
    deps='{:paths ['+' '.join(json.dumps(p) for p in case['paths'])+']}'
    (root/'deps.edn').write_text(deps,encoding='utf8')
    sources={}
    for name,marker in case['sources'].items():
        ns='other.namespace' if marker=='wrong_namespace' else case['ns']
        source=f'(ns {ns})\n(def source-marker "{marker}")'
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source,encoding='utf8')
        sources[name]=source
    client='client.'+platform;sources[client]=f'(ns client (:require [{case["ns"]}]))'
    (root/client).write_text(sources[client],encoding='utf8')
    graph,warnings=analyze_clojure(root,{n:{'language':'clojure'} for n in sources},{})
    candidates=graph[client]['platform_namespace_evidence'][platform]['candidates']
    return (candidates[0]['target_path'] if len(candidates)==1 else None,
            {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in [*sources,'deps.edn']},warnings)


def execute(root,case,platform,jars,java):
    if platform=='clj':
        expr=('(require (quote '+case['ns']+')) '
              '(if-let [n (find-ns (quote '+case['ns']+'))] '
              '(println "ATLAS_RESOURCE" (var-get (ns-resolve n (quote source-marker)))) '
              '(println "ATLAS_RESOURCE" "missing_namespace"))')
    else:
        expr=('(require (quote cljs.util)) '
              '(println "ATLAS_RESOURCE" (.getPath (cljs.util/ns->source (quote '+case['ns']+'))))')
    expr='(println "ATLAS_RUNTIME" (clojure-version)) '+expr
    cp=os.pathsep.join([*(str(p.resolve()) for p in jars),*(str(root/p) for p in case['paths'])])
    env={k:v for k,v in os.environ.items() if k in ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP')}
    proc=subprocess.run([java,'-Xmx384m','-Duser.home='+str(root),'-cp',cp,'clojure.main','-e',expr],
        cwd=root,env=env,text=True,capture_output=True,timeout=45,check=False)
    lines=[s.removeprefix('ATLAS_RESOURCE ') for s in proc.stdout.splitlines() if s.startswith('ATLAS_RESOURCE ')]
    if proc.returncode or len(lines)!=1:
        return {'status':'reference_process_error','exit_code':proc.returncode,'stderr_tail':proc.stderr[-800:]}
    versions=[s.removeprefix('ATLAS_RUNTIME ') for s in proc.stdout.splitlines() if s.startswith('ATLAS_RUNTIME ')]
    if len(versions)!=1 or (platform=='clj' and versions[0]!='1.12.6'):
        return {'status':'unexpected_reference_runtime','reported_versions':versions}
    value=lines[0]
    if platform=='cljs':
        path=Path(unquote(value)).resolve()
        if not path.is_relative_to(root):return {'status':'unexpected_reference_resource'}
        selected=path.relative_to(root).as_posix()
    elif value=='missing_namespace':selected=None
    else:
        matching=[p for p,marker in case['sources'].items() if marker==value]
        if len(matching)!=1:return {'status':'unexpected_reference_marker'}
        selected=matching[0]
    return {'status':'executed','selected_path':selected,'clojure_host_runtime':versions[0]}


def run(folder=None,cljs_jar=None,expected_cljs_sha256=None):
    report={'schema':'atlas-declared-source-path-reference/1','scope':'explicit resource order, not effective tools.deps basis',
        'project_code_executed':False,'automatic_downloads':False,'adapter_complete':False,
        'reference_runtime_executed':False,'cljs_reference_executed':False,
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256':hashlib.sha256(json.dumps([CASES,CLJS_CASES],sort_keys=True).encode()).hexdigest(),
        'cases':[]}
    jars=None;cljs=None;java=shutil.which('java')
    if folder is not None:
        jars=resolve_jar_dir(folder);report['jars']=preflight_jars(jars)
        if not java:raise ValueError('java not found')
    if cljs_jar is not None:
        evidence,cljs=_jar_preflight(cljs_jar,expected_cljs_sha256);report['cljs_preflight']=evidence
        if cljs is None:raise ValueError('CLJS expected digest preflight failed')
        if not java:raise ValueError('java not found')
    for platform,cases in [('clj',CASES),('cljs',CLJS_CASES)]:
        for case in cases:
            with tempfile.TemporaryDirectory(prefix='atlas-fixed-source-paths-') as tmp:
                root=Path(tmp);local,hashes,warnings=prepare(root,case,platform)
                row={'id':case['id'],'platform':platform,'expected':case['want'],'atlas':local,
                     'source_sha256':hashes,'local_passed':local==case['want'] and not warnings,
                     'reference_executed':False}
                runtime=jars if platform=='clj' else ([cljs] if cljs is not None else None)
                if runtime:
                    actual=execute(root,case,platform,runtime,java);row['reference']=actual
                    row['reference_executed']=True
                    row['matches']=row['local_passed'] and actual.get('status')=='executed' and actual.get('selected_path')==case['want']
                report['cases'].append(row)
    executed=[r for r in report['cases'] if r['reference_executed']]
    report.update(reference_runtime_executed=bool(executed),
        cljs_reference_executed=any(r['platform']=='cljs' for r in executed),
        reference_total=len(executed),reference_passed=sum(r['matches'] for r in executed),
        local_passed=sum(r['local_passed'] for r in report['cases']))
    return (0 if executed and all(r['matches'] for r in executed) else 2 if executed else 3),report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jar-dir',type=Path)
    p.add_argument('--cljs-jar',type=Path);p.add_argument('--expected-cljs-sha256')
    p.add_argument('--output',type=Path,default=Path('STAGE49_SOURCE_PATH_REFERENCE.json'));a=p.parse_args()
    code,report=run(a.jar_dir,a.cljs_jar,a.expected_cljs_sha256)
    if a.output.is_symlink():raise ValueError('report symlink refused')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('Source path reference:',report['reference_passed'],'/',report['reference_total'],'exit',code)
    return code
if __name__=='__main__':raise SystemExit(main())
