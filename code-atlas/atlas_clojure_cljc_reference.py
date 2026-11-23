#!/usr/bin/env python3
"""Stage40 opt-in Clojure JVM :clj projection of fixed .cljc fixture data.

NEVER sends source files of any project to a JVM. This only compares a bounded
structural data subset with *one* Clojure JVM version, not ClojureScript and
not macros/classpath/compilation/CFG/DFG. Zero automatic network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from analyzer.semantic_adapters.clojure_conditionals import project, ProjectionUnavailable
from analyzer.semantic_adapters.clojure_forms import lex, _Parser, ReaderError
from analyzer.semantic_adapters.clojure_shape import canonical_forms, UnsupportedShape
from atlas_clojure_reference import (resolve_jar_dir, preflight_jars, REFERENCE_VERSION)
from atlas_clojure_shape_reference import run_reference

POSITIVE = (
    ('single_selected', '#?(:clj [1 2] :cljs [3 4])'),
    ('selected_default', '#?(:cljr :other :default [4 5])'),
    ('no_matching_branch', '#?(:cljs [3 4])'),
    ('two_top_levels', '#?(:clj :yes) #?(:cljs :no) :last'),
    ('splice_vector', '[1 #?@(:clj [2 3] :cljs [4 5]) 6]'),
    ('splice_list', '(a #?@(:clj [b c] :cljs [d]) e)'),
    ('splice_map', '{:a 1 #?@(:clj [:b 2] :cljs [:c 3])}'),
    ('splice_empty', '[1 #?@(:clj [] :cljs [2]) 3]'),
    ('splice_unmatched', '[1 #?@(:cljs [2 3]) 4]'),
    ('nested_conditionals', '#?(:clj #?(:clj [1 2] :cljs [3]) :cljs [4])'),
    ('nested_splicing', '[:a #?@(:clj [:b #?@(:clj [:c :d])] :cljs [:e])]'),
    ('default_before_clj', '#?(:default :first :clj :second)'),
    ('namespace_require', '(ns demo.core (:require #?(:clj [clojure.string :as s] :cljs [cljs.string :as s])))'),
    ('namespace_require_splice', '(ns demo.core (:require #?@(:clj [[clojure.string :as s] [clojure.set :as set]] :cljs [[cljs.string :as s]])))'),
    ('map_value_conditional', '{:target #?(:clj :jvm :cljs :js)}'),
    ('unmatched_tail_ignored', '#?(:clj 1 :cljs)'),
    ('unmatched_tail_default', '#?(:default 1 :cljs)'),
    ('nonmatching_no_value', '#?(:cljs 1 :cljr)'),
    ('unselected_invalid_tag', '#?(:clj 1 cljs 2)'),
)
NEGATIVE = (
    ('splice_top_level', '#?@(:clj [1 2])'),
    ('splice_nonsequential', '[#?@(:clj 5)]'),
    ('missing_matched_value', '#?(:cljs 1 :clj)'),
    ('missing_first_matched', '#?(:clj)'),
    ('invalid_feature_before_match', '#?(clj 1 :clj 2)'),
    ('feature_not_keyword', '#?(clj 1 :cljs 2)'),
    ('duplicate_after_splice', '{:a 1 #?@(:clj [:a 2] :cljs [:b 2])}'),
    ('odd_map_after_omit', '{:a 1 #?@(:cljs [:b 2]) :c}'),
)
TASKS = POSITIVE + NEGATIVE


def local_case(name: str, source: str) -> dict:
    try:
        parsed = _Parser(lex(source,platform='shared')).all()
        result = project(parsed,'clj')
        shape = canonical_forms(result.forms)
        return {'id':name,'atlas_accepted':True,
                'atlas_forms':len(result.forms),
                'atlas_shape_sha256':hashlib.sha256(shape.encode('utf8')).hexdigest()}
    except (ReaderError,ProjectionUnavailable,UnsupportedShape,ValueError) as exc:
        return {'id':name,'atlas_accepted':False,'atlas_error':type(exc).__name__}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Stage40 explicit .cljc JVM reader-conditional fixture test')
    parser.add_argument('--jar-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-jars',action='store_true')
    parser.add_argument('--expected-clojure-sha256')
    parser.add_argument('--timeout',type=int,default=45)
    args=parser.parse_args(argv)
    if not 1<=args.timeout<=180:parser.error('timeout 1..180')
    if args.check_jars and not args.jar_dir:parser.error('--check-jars requires --jar-dir')
    if args.expected_clojure_sha256 and not args.jar_dir:parser.error('SHA256 requires --jar-dir')
    if args.expected_clojure_sha256 and not re.fullmatch('[0-9a-fA-F]{64}',args.expected_clojure_sha256):
        parser.error('SHA-256 must be 64 hex digits')
    good={name for name,_ in POSITIVE}
    rows=[dict(local_case(name,source),expected_accepted=name in good) for name,source in TASKS]
    report={'schema':'atlas-clojure-cljc-conditional-reference/1',
            'reference_release':REFERENCE_VERSION,
            'scope':'Explicit CLJC :clj reader-conditional selected data shapes, not full reader/AST/symbol resolution',
            'case_count':len(rows),'positive_count':len(POSITIVE),'negative_count':len(NEGATIVE),
            'fixture_sha256':hashlib.sha256(json.dumps(TASKS,ensure_ascii=False).encode()).hexdigest(),
            'local_passed':sum(r['atlas_accepted']==r['expected_accepted'] for r in rows),
            'reference_runtime_executed':False,'reference_binary_origin_attested':False,
            'project_code_executed':False,'network_requested':False,'cljs_certified':False,
            'semantic_or_forensic_certified':False,'complete_reader_conformance':False,'cases':rows}
    if args.jar_dir:
        jars=resolve_jar_dir(args.jar_dir)
        checked=preflight_jars(jars)
        report['preflight_jars']=[{'name':p.name,'sha256':x['sha256'],
                                   'publisher_signature_verified':False} for p,x in zip(jars,checked)]
        if args.expected_clojure_sha256 and checked[0]['sha256'].lower()!=args.expected_clojure_sha256.lower():
            raise ValueError('Clojure JAR checksum mismatch: JVM not started')
        report['user_supplied_digest_matched']=bool(args.expected_clojure_sha256)
        if not args.check_jars:
            harness=Path(__file__).resolve().parent/'reference'/'atlas_clojure_safe_conditional_reader.clj'
            version, refs=run_reference(jars,timeout=args.timeout,tasks=TASKS,harness=harness)
            report['reference_runtime_executed']=True;report['runtime_version']=version
            for r in rows:
                ref=refs[r['id']]
                r.update(ref)
                r['matches_reference']=(
                    r['atlas_accepted'] and ref['reference_accepted'] and
                    r['atlas_forms']==ref['reference_forms'] and
                    r['atlas_shape_sha256']==ref['reference_shape_sha256']
                ) if r['expected_accepted'] else (
                    not r['atlas_accepted'] and not ref['reference_accepted']
                    and ref['reference_forms']==0)
            report['matches_reference']=sum(r['matches_reference'] for r in rows)
    report['cljc_clj_subset_conformance_passed']=(report.get('matches_reference')==len(rows)
                                                     and report['reference_runtime_executed'])
    if args.output:
        if args.output.is_symlink():raise ValueError('Refuse report symlink')
        file=args.output.resolve();file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Atlas .cljc selected local: {report["local_passed"]}/{len(rows)}')
    if report['reference_runtime_executed']:
        print(f'Clojure JVM {report["runtime_version"]} .cljc: {report["matches_reference"]}/{len(rows)}')
        for r in rows:
            if not r['matches_reference']:
                print('MISMATCH:',r['id'],'local',r['atlas_accepted'],'jvm',r['reference_accepted'])
    else:print('JVM NEBYLA SPUŠTĚNA. Tímto výsledek konformity nevznikl.')
    if args.output:print('Report:',args.output)
    return 0 if report['cljc_clj_subset_conformance_passed'] else 2 if report['reference_runtime_executed'] else 3


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as exc:
        print('REFERENCE ERROR:',exc,file=sys.stderr)
        sys.exit(4)
