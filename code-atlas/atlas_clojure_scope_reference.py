#!/usr/bin/env python3
"""Opt-in fixed-fixture JVM lexical scope differential, never project execution.

Only embedded synthetic forms may be compiled/executed. No user source,
classpath, project directory, downloads, or user macros are accepted. This is
limited scope evidence, not Clojure/CLJS compiler or semantic certification.
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

from analyzer.semantic_adapters.clojure_forms import read_forms
from analyzer.semantic_adapters.clojure_scope import referred_call_sites
from atlas_clojure_reference import resolve_jar_dir, preflight_jars, REFERENCE_VERSION

# Calls carry their source line number as a trace label. JVM runs each truth
# branch separately; the UNION of reachable external calls is compared to the
# static candidates (which deliberately do not predict branch execution).
_IF = '(__MACRO__ [f (do\n (f 2)\n __VALUE__)]\n (when (fn? f) (f 4))\n (f 5))\n(f 6)'
_WHEN = '(__MACRO__ [f (do\n (f 2)\n __VALUE__)]\n (when (fn? f) (f 4)))\n(f 5)'
FIXTURES = tuple(
    (macro, (_IF if macro in ('if-let', 'if-some') else _WHEN).replace('__MACRO__', macro),
     ('(fn [& _] nil)', 'nil'), (2, 5, 6) if macro in ('if-let', 'if-some') else (2, 5))
    for macro in ('if-let', 'if-some', 'when-let', 'when-some', 'when-first')
) + (
    ('dotimes', _WHEN.replace('__MACRO__', 'dotimes'), ('1',), (2, 5)),
    ('with-open', _WHEN.replace('__MACRO__', 'with-open'),
     ('(java.io.StringReader. "fixture")',), (2, 5)),
    ('with-open-sequential', '(with-open [f (do\n (f 2)\n (java.io.StringReader. "a"))\n g (do (when (fn? f) (f 4)) (java.io.StringReader. "b"))]\n (when (fn? f) (f 5)))\n(f 6)',
     ('nil',), (2, 6)),
    ('shadowed-when-let', '(let [when-let (fn [x y] [x y])]\n (when-let [x 1] (f 3)))',
     (), ()),
)
# when-first needs a sequence whose first element is callable, not a function.
FIXTURES = tuple((name, source, ('[(fn [& _] nil)]', 'nil') if name == 'when-first'
                  else values, expected) for name, source, values, expected in FIXTURES)
_COMPREHENSION = '(doall (__MACRO__ [x (do\n (f 2)\n [1])\n :when (do (f 4) true)\n :let [f (do (f 5) (fn [& _] nil))]\n :while (do (when (fn? f) (f 6)) true)\n y [f]] (f 7)))\n(f 8)'
FIXTURES += tuple((macro+'-modifiers', _COMPREHENSION.replace('__MACRO__',macro),
                   ('nil',), (2,4,5,8)) for macro in ('for','doseq')) + (
    ('for-destructuring', '(doall (for [[f] (do\n (f 2)\n [[(fn [& _] nil)]])]\n (f 4)))\n(f 5)',
     ('nil',), (2,5)),
    ('as-thread', '(as-> (do\n (f 2)\n (fn [& _] nil)) f\n (f 4)\n (when (fn? f) (f 5)))\n(f 6)',
     ('nil',), (2,6)),
)

# Stage50: qualified public core macros must preserve lexical identity. The
# function objects in these fixed fixtures do not come from project sources.
FIXTURES += tuple(
    ('qualified-'+macro,
     '(clojure.core/'+macro+' [f (do\n (f 2)\n (fn [& _] nil))]\n (f 4))\n(f 5)',
     ('nil',), (2, 5)) for macro in ('let', 'loop')
) + (
    ('qualified-fn', '(clojure.core/fn [f] (f 1))\n(f 2)', ('nil',), (2,)),
    ('qualified-letfn', '(clojure.core/letfn [(f [& _] nil)]\n (f 2))\n(f 3)',
     ('nil',), (3,)),
    ('qualified-case', '(clojure.core/case\n (do (f 2) __VALUE__)\n (f 3)\n (f 4)\n (f 5))',
     # case list tests group constants f and 3; dispatch 3 reaches that arm.
     ('3', 'nil'), (2, 4, 5)),
    ('qualified-comment', '(clojure.core/comment (f 1))\n(f 2)', ('nil',), (2,)),
)


def baseline_report():
    cases=[]
    for name, source, values, expected in FIXTURES:
        concrete=source.replace('__VALUE__', values[0] if values else 'nil')
        hits, omissions=referred_call_sites(tuple(read_forms(concrete)), frozenset({'f'}))
        actual=sorted({hit.head.line for hit in hits})
        cases.append({'id':name, 'static_external_call_lines':actual,
                      'expected_external_call_lines':list(expected),
                      'local_passed':actual==list(expected),
                      'omission_reasons':sorted({s.reason for s in omissions})})
    return {'schema':'atlas-clojure-scope-reference/1', 'reference_release':REFERENCE_VERSION,
            'scope':'Fixed JVM lexical binding fixtures only; no CLJS/compiler conformance claim',
            'case_count':len(cases), 'local_baseline_passed':sum(c['local_passed'] for c in cases),
            'corpus_sha256':hashlib.sha256(json.dumps(FIXTURES).encode()).hexdigest(),
            'reference_runtime_executed':False, 'reference_binary_origin_attested':False,
            'scope_subset_matches_reference':False, 'project_code_executed':False,
            'network_requested':False, 'complete_semantic_conformance':False,
            'forensic_certified':False, 'cases':cases}


def run_reference(jars, timeout=45):
    java=shutil.which('java')
    if not java:raise RuntimeError('Java not available')
    # No source input argument: all executable text is assembled from FIXTURES.
    expressions=['(println (str "VERSION " (clojure-version)))']
    for name, source, values, _ in FIXTURES:
        if not values:
            # Ambiguous user macro/function is intentionally suppressed locally;
            # it is outside the fixed core-macro runtime differential subset.
            continue
        for index,value in enumerate(values):
            concrete=source.replace('__VALUE__', value)
            expressions.append('(let [trace (atom []) f (fn [line] (swap! trace conj line) nil)] '
                               + concrete + '\n(println "CASE '+name+' '+str(index)+'" (pr-str @trace)))')
    code='\n'.join(expressions)
    env={key:os.environ[key] for key in ('PATH','SYSTEMROOT','WINDIR') if key in os.environ}
    with tempfile.TemporaryDirectory(prefix='atlas-clojure-fixed-scope-') as cwd:
        proc=subprocess.run([java,'-cp',os.pathsep.join(str(Path(j).resolve()) for j in jars),
                             'clojure.main','-e',code],cwd=cwd,env=env,text=True,
                            capture_output=True,timeout=timeout,check=False)
    if proc.returncode:raise RuntimeError('Fixed scope JVM fixture failed: '+proc.stderr[:1200])
    lines=proc.stdout.splitlines()
    if not lines or lines[0]!='VERSION '+REFERENCE_VERSION:
        raise RuntimeError('Unexpected Clojure runtime version or reference output')
    results={}
    for line in lines[1:]:
        match=re.fullmatch(r'CASE ([a-z-]+) (\d+) \[([\d ]*)\]',line)
        if not match:raise RuntimeError('Unexpected scope fixture output')
        key=(match[1],int(match[2]))
        if key in results:raise RuntimeError('Duplicate scope fixture output')
        results[key]=[int(x) for x in match[3].split()]
    expected={(name,i) for name,_,values,_ in FIXTURES for i in range(len(values))}
    if set(results)!=expected:raise RuntimeError('Missing or unexpected scope fixtures')
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jar-dir',type=Path)
    parser.add_argument('--check-jars',action='store_true')
    parser.add_argument('--timeout',type=int,default=45)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not 1<=args.timeout<=180:parser.error('--timeout must be 1..180')
    if args.check_jars and not args.jar_dir:parser.error('--check-jars needs --jar-dir')
    report=baseline_report()
    if args.jar_dir:
        jars=resolve_jar_dir(args.jar_dir)
        checks=preflight_jars(jars)
        report['preflight_jars']=[{'name':p.name,'sha256':r['sha256'],
                                  'publisher_signature_verified':False} for p,r in zip(jars,checks)]
        if not args.check_jars:
            results=run_reference(jars,args.timeout)
            report['reference_runtime_executed']=True
            compared=[]
            for row,(_,_,values,_) in zip(report['cases'],FIXTURES):
                if not values:
                    row['reference_not_run_reason']='Ambiguous macro/function kept opaque'
                    continue
                observed=sorted({line for i in range(len(values)) for line in results[(row['id'],i)]})
                row['reference_external_call_lines']=observed
                row['matches_reference']=observed==row['static_external_call_lines']==row['expected_external_call_lines']
                compared.append(row['matches_reference'])
            report['reference_case_count']=len(compared)
            report['matches_reference']=sum(compared)
            report['scope_subset_matches_reference']=bool(compared) and all(compared)
    if not report['reference_runtime_executed']:
        report['reference_not_run_reason']='No local JARs supplied or preflight only'
    if args.output:
        if args.output.is_symlink():raise ValueError('Output symlink refused')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return (0 if report['scope_subset_matches_reference'] else 2) if report['reference_runtime_executed'] else 3


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit('Scope reference failed: '+str(exc))
