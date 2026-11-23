#!/usr/bin/env python3
"""Read-only Atlas Stage42 semantic regression reproducer, no user project execution.

Usage: py -3 AUDIT_STAGE42_REPRO.py "C:\\path\\to\\Universal-Code-Atlas-Stage42"
All source samples are synthetic and are created in a system temp directory.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def analyze(samples, analyze_clojure):
    with TemporaryDirectory(prefix='atlas42-guard-audit-') as temp:
        root=Path(temp)
        for file, text in samples.items():
            target=root/file;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(text,encoding='utf-8')
        return analyze_clojure(root,{p:{'language':'clojure'} for p in samples},{})


def main():
    parser=argparse.ArgumentParser(description='Stage42 semantic guard audit: only synthetic temp fixtures')
    parser.add_argument('atlas_root',type=Path,help='Main directory of extracted Atlas Stage42')
    args=parser.parse_args()
    root=args.atlas_root.resolve()
    if not (root/'analyzer/semantic_adapters/clojure_forms.py').is_file():
        parser.error('Please provide an extracted Atlas root containing analyzer/semantic_adapters/clojure_forms.py')
    sys.path.insert(0,str(root))
    from analyzer.semantic_adapters.clojure_forms import analyze_clojure
    private={
      'lib.clj':'(ns lib.core)\n(defn- hidden [] :secret)\n(defn ^:private also-hidden [] :secret)\n(defn public [] :ok)\n',
      'app.clj':'(ns app.core (:require [lib.core :as l]))\n(defn run [] (l/hidden) (l/also-hidden) (l/public))\n'
    }
    graphs,_=analyze(private,analyze_clojure)
    call_edges=[e for e in graphs['app.clj']['edges'] if e.get('relation')=='clojure_call_candidate']
    print('PRIVATE FIXTURE: candidate calls:',len(call_edges),'(expected ordinary public call only: 1)')
    bad_private=len(call_edges)>1
    collision={
      'dep.clj':'(ns lib.core)\n(defn get-value [] 1)\n',
      'other.clj':'(ns another.core)\n(defn get-value [] 2)\n',
      'app.clj':'(ns app.core (:require [lib.core :as x] [another.core :as-alias x]))\n(defn run [] (x/get-value))\n'
    }
    graphs,_=analyze(collision,analyze_clojure)
    candidate=[e for e in graphs['app.clj']['edges'] if e.get('relation')=='clojure_call_candidate']
    print('COLLIDING ALIAS FIXTURE: candidate calls:',len(candidate),'(desired fail-closed: 0)')
    bad_collision=bool(candidate)
    if bad_private or bad_collision:
        print('AUDIT RESULT: SEMANTIC GUARD ISSUES REPRODUCED')
        return 2
    print('AUDIT RESULT: THESE TWO GUARDS APPEAR RESOLVED IN THIS BUILD')
    return 0


if __name__=='__main__':
    sys.exit(main())
