#!/usr/bin/env python3
"""Opt-in fixed Clojure JVM oracle for explicit CLI alias *path* scenarios.

Only own bounded synthetic files are created; never accepts repo/source or
analyzed project arguments. Runtime is opt-in through --jar-dir AND --run,
without Clojure CLI, network, Maven, tools.deps or external project code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from analyzer.semantic_adapters.clojure_forms import analyze_clojure
from atlas_clojure_reference import resolve_jar_dir, preflight_jars

CASES=(
    ('default_no_alias', (), None, 'src'),
    ('extra_precedes_project', (':test',), ':test', 'test'),
    ('later_extra_wins', (':test', ':dev'), ':test:dev', 'dev'),
    ('replace_project_path', (':replace',), ':replace', 'test'),
    ('later_replace_with_extra', (':dev', ':replace', ':back'), ':dev:replace:back','dev'),
)
DEPS='{:paths ["src"] :aliases {:test {:extra-paths ["test"]} :dev {:extra-paths ["dev"]} :replace {:replace-paths ["test"]} :back {:replace-paths ["src"]}}}'
SOURCES={
    'src/example/shared.clj':'(ns example.shared) (def source-marker "src")',
    'test/example/shared.clj':'(ns example.shared) (def source-marker "test")',
    'dev/example/shared.clj':'(ns example.shared) (def source-marker "dev")',
    'src/app.clj':'(ns app (:require [example.shared]))',
    'test/app.clj':'(ns app (:require [example.shared]))',
}


def check_case(case,root,jars=None,java=None):
    case_id,selected,cli,expected=case
    for name,content in {'deps.edn':DEPS,**SOURCES}.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf8')
    graphs,warnings=analyze_clojure(root,{name:{'language':'clojure'} for name in SOURCES},
                                   {'_atlas_clojure_cli_aliases':selected})
    client='test/app.clj' if case_id=='replace_project_path' else 'src/app.clj'
    evidence=graphs[client]['platform_namespace_evidence']['clj']
    paths=evidence['candidates']
    target=paths[0]['target_path'] if len(paths)==1 else None
    row={'id':case_id,'cli_aliases':cli or '', 'selected_cli_aliases':list(selected),
         'expected_target':expected+'/example/shared.clj', 'atlas_target':target,
         'atlas_passed':target==expected+'/example/shared.clj' and not warnings and
                        not paths[0]['semantic_resolution_certified'] if paths else False,
         'reference_executed':False,'reference_passed':False,
         'manifest_sha256':hashlib.sha256(DEPS.encode('utf8')).hexdigest()}
    if jars is None:return row
    classpath=os.pathsep.join([*(str(p.resolve()) for p in jars),
        *(str(root/entry['path']) for entry in graphs[client]['clojure_project_context']['declared_roots'])])
    expression='(require (quote example.shared)) (println "ATLAS_ALIAS_REFERENCE" example.shared/source-marker)'
    environment={k:v for k,v in os.environ.items() if k in ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP')}
    proc=subprocess.run([java,'-Xmx384m','-Duser.home='+str(root),'-cp',classpath,
        'clojure.main','-e',expression],cwd=root,env=environment,
        capture_output=True,text=True,timeout=40,check=False)
    markers=[line.split('ATLAS_ALIAS_REFERENCE ',1)[1] for line in proc.stdout.splitlines()
             if 'ATLAS_ALIAS_REFERENCE ' in line]
    row.update(reference_executed=True,reference_returncode=proc.returncode,
               reference_markers=markers,reference_passed=(proc.returncode==0 and markers==[expected]),
               reference_stderr_tail=proc.stderr[-220:] if proc.returncode else '')
    return row


def run(jar_dir=None,explicit_run=False):
    report={'schema':'atlas-stage53-fixed-cli-alias-path-oracle/1','scope':'static declared path scenario, NOT effective tools.deps basis',
            'alias_precedence_reference':'https://clojure.org/reference/clojure_cli',
            'project_source_executed':False,'network_access_required':False,
            'auto_downloaded':False,'compiler_expansion_executed':False,'adapter_complete':False,
            'reference_runtime_executed':False,'cases':[]}
    jars=None;java=None
    if explicit_run:
        if jar_dir is None:raise ValueError('--run requires explicit --jar-dir')
        jars=resolve_jar_dir(Path(jar_dir))
        report['jars']=preflight_jars(jars)
        java=shutil.which('java')
        if not java:raise ValueError('explicit reference run requires locally installed Java')
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix='atlas-fixed-clj-alias-') as temp:
            report['cases'].append(check_case(case,Path(temp),jars,java))
    report.update(local_passed=sum(r['atlas_passed'] for r in report['cases']),
                  reference_passed=sum(r['reference_passed'] for r in report['cases']),
                  reference_total=sum(r['reference_executed'] for r in report['cases']),
                  reference_runtime_executed=explicit_run)
    return (0 if all(r['atlas_passed'] for r in report['cases']) and
                (not explicit_run or report['reference_passed']==len(CASES)) else 2),report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jar-dir',type=Path)
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--output',type=Path,default=Path('STAGE53_CLOJURE_ALIAS_REFERENCE.json'))
    args=ap.parse_args()
    if args.output.is_symlink():raise ValueError('output symlink refused')
    status,result=run(args.jar_dir,args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Stage53 alias path oracle:',result['local_passed'],'/',len(CASES),
          'JVM:',result['reference_passed'],'/',result['reference_total'])
    return status

if __name__=='__main__':raise SystemExit(main())
