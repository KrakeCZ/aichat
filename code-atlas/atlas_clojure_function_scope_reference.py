#!/usr/bin/env python3
"""Opt-in JVM 1.12.6 function-scope differential on fixed embedded forms only.

The JVM evaluates only this script's fixtures, never an analyzed project or a
caller-supplied expression. No network, compiler downloads or project classpath.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from analyzer.semantic_adapters.clojure_forms import read_forms, Form
from analyzer.semantic_adapters.clojure_scope import referred_call_sites
from atlas_clojure_reference import REFERENCE_VERSION, resolve_jar_dir, preflight_jars

# Expected values are source lines. Each external call passes its literal line
# to the logger; duplicates are retained, so two calls on one line cannot mask
# an incorrectly inferred local call. Runtime order is not a static CFG claim.
FIXTURES=(
    ('pre-body-post', '((fn []\n {:pre [(f 2)]\n :post [(% 3) (g 3)]}\n (% 4)))', (2,3,4), False),
    ('final-map-value', '((fn []\n {:post [(% 2)] :other (g 2)}))', (2,2), False),
    ('metadata-contracts', '((fn ^{:pre [(f 1)] :post [(% 1)]} []\n (% 2)))', (1,2), False),
    ('map-overrides-metadata', '((fn ^{:pre [(f 1)] :post [(g 1)]} [] {}\n (% 2)))', (2,), False),
    ('unused-contract-data', '((fn []\n {:pre [(f 2)] :post [(% 2)] :unused (g 2)}\n (% 3)))', (2,3), False),
    ('letfn-contracts', '(letfn [(local []\n {:pre [(f 2)] :post [(% 2)]}\n (% 3))]\n (local))', (2,3), False),
    ('raw-fn-star-map', '((fn* []\n {:post [(% 2)] :other (g 2)}\n (f 3)))', (2,2,3), False),
    ('nested-body-closure', '(((fn []\n {:post [(ifn? %)]}\n (fn [] (% 3)))))', (3,), False),
    ('nil-false-conditions', '((fn [] {:pre nil :post false}\n (f 2)))', (2,), False),
    ('literal-condition-list', '((fn [] {:pre (f)}\n (g 2)))', (2,), False),
    ('invalid-pre', '(fn [] {:pre true} (g 1))', (), False),
    ('invalid-post', '(fn [] {:post 42} (g 1))', (), False),
    ('macro-implicit-locals', '(defmacro fixed-m []\n (when (ifn? &env) (&env 2))\n (when (ifn? &form) (&form 3))\n (f 4))\n(macroexpand (quote (fixed-m)))', (4,), True),
)
NEGATIVES={'invalid-pre','invalid-post'}


def baseline_report():
    rows=[]
    for name,source,expected,macro in FIXTURES:
        forms=tuple(read_forms(source))
        if macro:
            declaration=forms[0]
            head=Form('internal_function_scope','fn*',(),declaration.line,declaration.col)
            forms=(Form('list','(', (head,*declaration.children[2:]),declaration.line,declaration.col),)
        hits,omissions=referred_call_sites(forms,frozenset({'f','g','%','&env','&form'}),
            implicit_bindings=frozenset({'&env','&form'}) if macro else frozenset())
        actual=sorted(h.head.line for h in hits)
        rows.append({'id':name,'expected_call_lines':list(expected),'static_call_lines':actual,
            'expected_rejection':name in NEGATIVES,'omission_reasons':sorted({s.reason for s in omissions}),
            'local_passed':actual==list(expected) and (name not in NEGATIVES or
                any(s.reason=='unsupported_function_conditions' for s in omissions))})
    return {'schema':'atlas-function-scope-reference/1','reference_release':REFERENCE_VERSION,
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256':hashlib.sha256(json.dumps(FIXTURES).encode()).hexdigest(),
        'scope':'Fixed fn contracts and JVM defmacro implicit locals only',
        'reference_runtime_executed':False,'user_project_code_executed':False,
        'automatic_downloads':False,'adapter_complete':False,'cases':rows,
        'local_passed':sum(row['local_passed'] for row in rows),'case_count':len(rows)}


def run_reference(folder,timeout=45):
    report=baseline_report();jars=resolve_jar_dir(folder)
    report['jars']=preflight_jars(jars)
    java=shutil.which('java')
    if not java:raise ValueError('Java not available')
    code=['(ns atlas.stage51.fixed)', '(println "VERSION" (clojure-version))',
          '(def trace (atom []))', '(defn f [line] (swap! trace conj line) true)',
          '(defn g [line] (swap! trace conj line) true)',
          '(defn % [line] (swap! trace conj line) (fn [& _] true))',
          '(defn &env [line] (swap! trace conj line) true)',
          '(defn &form [line] (swap! trace conj line) true)']
    for name,source,_,_ in FIXTURES:
        code.append('(reset! trace []) (try (eval (quote (do '+source+')))' +
            ' (println "CASE '+name+' OK" (pr-str @trace))'+
            ' (catch Throwable e (println "CASE '+name+' REJECT" (pr-str @trace))))')
    env={k:v for k,v in os.environ.items() if k in ('PATH','SYSTEMROOT','SystemRoot','WINDIR','TEMP','TMP')}
    with tempfile.TemporaryDirectory(prefix='atlas-fixed-function-scope-') as tmp:
        root=Path(tmp);script=root/'fixed.clj';script.write_text('\n'.join(code),encoding='utf-8')
        proc=subprocess.run([java,'-Xmx384m','-Duser.home='+str(root),'-cp',
            os.pathsep.join(str(p.resolve()) for p in jars),'clojure.main',str(script)],
            cwd=root,env=env,capture_output=True,text=True,timeout=timeout,check=False)
    report['reference_process_attempted']=True
    if proc.returncode:
        report.update(reference_error='reference_process_failed',stderr_head=proc.stderr[:1800])
        return 2,report
    if 'VERSION '+REFERENCE_VERSION not in proc.stdout.splitlines():
        report['reference_error']='unexpected_runtime_version';return 2,report
    observed={}
    for line in proc.stdout.splitlines():
        match=re.fullmatch(r'CASE ([a-z-]+) (OK|REJECT) \[([\d ]*)\]',line)
        if match:
            if match[1] in observed:raise ValueError('duplicate fixture output')
            observed[match[1]]=(match[2],sorted(int(n) for n in match[3].split()))
    if set(observed)!={r['id'] for r in report['cases']}:raise ValueError('missing fixture output')
    report['reference_runtime_executed']=True
    positives_ok=all(observed[r['id']][0]=='OK' and observed[r['id']][1]==r['expected_call_lines']
                     for r in report['cases'] if not r['expected_rejection'])
    for row in report['cases']:
        state,lines=observed[row['id']]
        row.update(reference_state=state,reference_call_lines=lines,
            matches_reference=row['local_passed'] and lines==row['expected_call_lines'] and
            ((state=='REJECT' and positives_ok) if row['expected_rejection'] else state=='OK'))
    report['reference_passed']=sum(r['matches_reference'] for r in report['cases'])
    report['reference_total']=len(report['cases'])
    return (0 if report['reference_passed']==report['reference_total'] else 2),report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jar-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('FUNCTION_SCOPE_REFERENCE_RESULT.json'))
    args=parser.parse_args()
    code,report=run_reference(args.jar_dir) if args.jar_dir else (3,baseline_report())
    if args.output.is_symlink():raise ValueError('output symlink refused')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Function scope:',report.get('reference_passed',0),'/',report.get('reference_total',0),
          'JVM executed:',report['reference_runtime_executed'],'exit',code)
    return code


if __name__=='__main__':raise SystemExit(main())
