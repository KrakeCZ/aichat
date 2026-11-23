#!/usr/bin/env python3
"""Opt-in Stage47 fixed compiler controls; no user-project paths are accepted."""
import argparse
from pathlib import Path
from atlas_cljs_compiler_reference import Fixture, run_reference

MARKER='ATLAS_STAGE47_MACRO_EXPANDED_73E91'
MACROS=f'(ns atlas.oracle.macros)\n(defmacro stamp [] "{MARKER}")\n'
RUNTIME='(ns atlas.oracle.macros (:require-macros atlas.oracle.macros))\n(defn plain [] 2)\n'
ORIGIN=':require/implicit-self-macro-loading'
FIXTURES=(
 Fixture('implicit_alias_expansion',
  '(ns atlas.oracle.client (:require [atlas.oracle.macros :as m]))\n(def result (m/stamp))',
  provider=MACROS,runtime_lib=RUNTIME,required_refer=(),expected_decl_status=None,
  origin=ORIGIN,expected_js_marker=MARKER),
 Fixture('implicit_refer_expansion',
  '(ns atlas.oracle.client (:require [atlas.oracle.macros :refer [stamp plain]]))\n(def result (stamp))\n(def value (plain))',
  provider=MACROS,runtime_lib=RUNTIME,required_refer=('stamp',),
  origin=ORIGIN,expected_js_marker=MARKER),
 Fixture('explicit_private_macro_expansion',
  '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [stamp]]))\n(def result (stamp))',
  provider=MACROS.replace('defmacro stamp','defmacro ^:private stamp'),required_refer=('stamp',),
  expected_decl_status='private_direct_defmacro_not_public_candidate',expected_js_marker=MARKER),
 Fixture('missing_explicit_macro_negative',
  '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [missing]]))\n(def result (missing))',
  provider=MACROS,required_refer=('missing',),
  expected_decl_status='no_literal_defmacro_found_not_proof_of_absence',compiler_expectation='compile_failure'),
)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--cljs-jar',type=Path)
    p.add_argument('--expected-sha256');p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--output',type=Path,default=Path('STAGE47_COMPILER_REFERENCE.json'))
    args=p.parse_args()
    code,report=run_reference(jar=args.cljs_jar,expected_sha256=args.expected_sha256,
        run=args.run,timeout=args.timeout,output=None,fixtures=FIXTURES)
    report['compiler_conformance_scope']='stage47_fixed_implicit_macro_loading_and_emitted_expansion_markers'
    report['fixture_source']='atlas_cljs_implicit_reference.py'
    # Reuse the guarded output behavior of the base harness at the CLI boundary.
    import json
    if args.output.is_symlink():
        raise ValueError('refuse report symlink')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Stage47 compiler:',report['reference_passed_count'],'positive,',
          report.get('reference_negative_passed_count',0),'negative passed; exit',code)
    return code

if __name__=='__main__':
    raise SystemExit(main())
