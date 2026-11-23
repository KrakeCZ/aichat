#!/usr/bin/env python3
"""Opt-in JVM 1.12.6 oracle for fixed protocol, record and type fixtures.

Never loads caller source files, build scripts, or the analyzed project.
The only executable source is the hardcoded, reviewed FIXTURE below.
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

from atlas_clojure_reference import resolve_jar_dir, preflight_jars
from analyzer.semantic_adapters.clojure_forms import analyze_clojure

FIXTURE='''(ns atlas.stage52.members)
(defprotocol Shape (area [this] [this scale] "documented"))
(defrecord Point [x y] Shape
  (area [this] x)
  (area [this scale] (* x scale)))
(deftype Plain [value])
(println "ATLAS_PROTO_VAR" (contains? (ns-publics 'atlas.stage52.members) 'area))
(println "ATLAS_RECORD_FIELDS" (every? (set (map #(.getName %) (.getDeclaredFields Point))) ["x" "y"]))
(println "ATLAS_TYPE_FIELD" (contains? (set (map #(.getName %) (.getDeclaredFields Plain))) "value"))
(println "ATLAS_DISPATCH_OBSERVED" (area (->Point 7 2) 3))
'''
EXPECTED={'ATLAS_PROTO_VAR':'true','ATLAS_RECORD_FIELDS':'true',
          'ATLAS_TYPE_FIELD':'true','ATLAS_DISPATCH_OBSERVED':'21'}


def run(*, jar_dir=None, execute=False, timeout=40):
    if not 5<=timeout<=120:
        raise ValueError('timeout outside safe range')
    with tempfile.TemporaryDirectory(prefix='atlas-stage52-reviewed-fixtures-') as directory:
        root=Path(directory)
        src=root/'atlas/stage52/members.clj'
        src.parent.mkdir(parents=True)
        src.write_text(FIXTURE,encoding='utf8')
        graphs,warnings=analyze_clojure(root,{'atlas/stage52/members.clj':{'language':'clojure'}},{})
        graph=graphs['atlas/stage52/members.clj']
        kinds={(n['name'],n['kind']) for n in graph['nodes']}
        expected_kinds={('Shape','protocol'),('area','protocol_method_declaration'),
                        ('Point','record'),('x','record_field_declaration'),
                        ('y','record_field_declaration'),('Plain','type'),
                        ('value','type_field_declaration')}
        static_pass=expected_kinds.issubset(kinds) and not warnings and not graph['declaration_limitations']
        report={'schema':'atlas-stage52-member-reference/1',
                'official_target_jvm':'1.12.6','fixture_sha256':hashlib.sha256(FIXTURE.encode()).hexdigest(),
                'local_syntax_passed':static_pass,'observed_declared_symbols':sorted(kinds),
                'reference_executed':False,'reference_compiler_conformance_complete':False,
                'user_project_executed':False,'automatic_network_downloads':False,
                'dispatch_is_statically_resolved':False,'warnings':warnings}
        if not execute:
            report['status']='not_run_requires_explicit_flag'
            return 3,report
        if jar_dir is None:
            report['status']='not_run_missing_explicit_jars'
            return 3,report
        jars=resolve_jar_dir(jar_dir)
        report['jar_verification']=preflight_jars(jars)
        java=shutil.which('java')
        if java is None:
            report['status']='not_run_java_missing'
            return 3,report
        if not static_pass:
            report['status']='not_run_static_baseline_failed'
            return 2,report
        env={k:v for k,v in os.environ.items() if k in
             ('PATH','SYSTEMROOT','SystemRoot','WINDIR','TEMP','TMP')}
        proc=subprocess.run([java,'-Xmx384m','-Duser.home='+str(root),'-cp',
            os.pathsep.join(str(x.resolve()) for x in jars),'clojure.main',str(src)],
            cwd=root,env=env,capture_output=True,text=True,timeout=timeout,check=False)
        report['reference_executed']=True
        report['reference_exit_code']=proc.returncode
        observed={}
        for line in proc.stdout.splitlines():
            head,_,value=line.partition(' ')
            if head in EXPECTED:
                if head in observed:
                    report['status']='duplicate_oracle_marker'
                    return 2,report
                observed[head]=value
        report['expected_reference_markers']=EXPECTED
        report['observed_reference_markers']=observed
        if proc.returncode or observed!=EXPECTED:
            report['status']='reference_failed_or_inconclusive'
            report['stderr_head']=proc.stderr[:1400]
            return 2,report
        report['status']='fixed_subset_passed'
        return 0,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--jar-dir',type=Path)
    p.add_argument('--timeout',type=int,default=40)
    p.add_argument('--output',type=Path,default=Path('STAGE52_MEMBERS_REFERENCE.json'))
    args=p.parse_args()
    status,report=run(jar_dir=args.jar_dir,execute=args.run,timeout=args.timeout)
    if args.output.is_symlink():
        raise ValueError('refuse symlink output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'], 'static=',report['local_syntax_passed'],'reference=',report['reference_executed'])
    return status


if __name__=='__main__':
    raise SystemExit(main())
