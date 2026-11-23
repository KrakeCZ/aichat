#!/usr/bin/env python3
"""Opt-in Clojure 1.12.6 numeric reader equality fixtures, no project access.

Source acceptance and canonical data-shape comparison are deliberately kept
separate from any semantics/AST/forensic certification. A real JVM run only
occurs after the user supplies 3 local checked JARs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from atlas_clojure_reference import resolve_jar_dir, preflight_jars, REFERENCE_VERSION
from atlas_clojure_shape_reference import _local_case, run_reference

# Officially documented equivalent integral literals, and duplicate keys/sets.
# No macros, eval, user classpath or user-supplied source are accepted.
NUMERIC_POSITIVE = (
    ('equivalent_integer_bases', '[42 052 0x2a 2r101010 36r16 8r52]'),
    ('negative_integer_bases', '[-42 -052 -0x2a -2r101010]'),
    ('radix_and_keyword_map', '{:decimal 42 :hex 0x2a :octal 052 :base2 2r101010}'),
    ('nested_numeric_map', '{[0x2a 052] #{2r101010 36r17}}'),
    ('binary_and_hex_distinct', '[0x2b 2r101010 0x2a]'),
    ('bigint_base_equivalence', '[42N 42 0x2a]'),
)
NUMERIC_REJECT = (
    ('duplicate_octal_decimal_set', '#{42 052}'),
    ('duplicate_hex_decimal_map', '{0x2a :hex 42 :decimal}'),
    ('duplicate_radix_values_set', '#{2r101010 36r16}'),
    ('duplicate_radix_hex_map', '{8r52 :oct 0x2a :hex}'),
)
NUMERIC_TASKS = NUMERIC_POSITIVE + NUMERIC_REJECT


def baseline():
    good={name for name,_ in NUMERIC_POSITIVE}
    return [dict(_local_case(name,source),expected_accepted=name in good)
            for name,source in NUMERIC_TASKS]


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage39: oddělené ověření číselných literálů Clojure proti JVM')
    parser.add_argument('--jar-dir', type=Path)
    parser.add_argument('--expected-clojure-sha256')
    parser.add_argument('--check-jars', action='store_true')
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args(argv)
    if not 1<=args.timeout<=180:parser.error('timeout musí být 1 až 180')
    if args.check_jars and not args.jar_dir:parser.error('--check-jars vyžaduje --jar-dir')
    if args.expected_clojure_sha256 and not args.jar_dir:parser.error('Digest vyžaduje --jar-dir')
    if args.expected_clojure_sha256 and not re.fullmatch('[0-9A-Fa-f]{64}',args.expected_clojure_sha256):
        parser.error('SHA-256 musí mít 64 znaků')
    cases=baseline()
    report={
        'schema':'atlas-clojure-numeric-reference/1',
        'reference_release':REFERENCE_VERSION,
        'scope':'Only fixed numeric data-reader fixtures; no AST, macros or semantic certification',
        'case_count':len(cases),
        'corpus_sha256': hashlib.sha256(json.dumps(NUMERIC_TASKS,ensure_ascii=False).encode()).hexdigest(),
        'local_baseline_passed':sum(row['local_accepted']==row['expected_accepted'] for row in cases),
        'reference_runtime_executed':False,
        'reference_binary_origin_attested':False,
        'project_code_executed':False,
        'network_requested':False,
        'complete_reader_conformance':False,
        'forensic_certified':False,
        'cases':cases,
    }
    if args.jar_dir:
        jars=resolve_jar_dir(args.jar_dir)
        checked=preflight_jars(jars)
        report['preflight_passed']=True
        report['preflight_jars']=[{'name':path.name,'sha256':row['sha256'],
                                   'publisher_signature_verified':False}
                                  for path,row in zip(jars,checked)]
        if args.expected_clojure_sha256 and checked[0]['sha256'].lower()!=args.expected_clojure_sha256.lower():
            raise ValueError('SHA-256 JARu nesouhlasí, JVM nebyla spuštěna')
        report['user_supplied_digest_matched']=bool(args.expected_clojure_sha256)
        if not args.check_jars:
            version, results=run_reference(jars,timeout=args.timeout,tasks=NUMERIC_TASKS)
            report['reference_runtime_executed']=True
            report['runtime_version']=version
            for row in cases:
                ref=results[row['id']]
                row.update(ref)
                if row['expected_accepted']:
                    row['matches_reference']=(row['local_accepted'] and ref['reference_accepted'] and
                      row['local_forms']==ref['reference_forms'] and
                      row['local_shape_sha256']==ref['reference_shape_sha256'])
                else:
                    row['matches_reference']=(not row['local_accepted'] and
                        not ref['reference_accepted'] and ref['reference_forms']==0)
            report['matches_reference']=sum(row['matches_reference'] for row in cases)
            report['numeric_subset_conformance_passed']=(report['matches_reference']==len(cases))
    if not report['reference_runtime_executed']:
        report['numeric_subset_conformance_passed']=False
        report['reference_not_run_reason']='No JARs or preflight only'
    if args.output:
        if args.output.is_symlink():raise ValueError('Output symlink refused')
        out=args.output.resolve()
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Atlas numeric local: {report["local_baseline_passed"]}/{len(cases)}')
    if report['reference_runtime_executed']:
        print(f'Clojure JVM {report["runtime_version"]}: {report["matches_reference"]}/{len(cases)}')
    else:print('REFERENCE JVM NEBYLA SPUŠTĚNA: není možné potvrdit shodu s oficiálním readerem.')
    if args.output:print('Report:',args.output)
    if not report['reference_runtime_executed']:return 3
    return 0 if report['numeric_subset_conformance_passed'] else 2


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError) as err:
        print('CHYBA REFERENČNÍ KONTROLY:',err,file=sys.stderr)
        sys.exit(4)
