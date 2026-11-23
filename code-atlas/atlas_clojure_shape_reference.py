#!/usr/bin/env python3
"""Explicit opt-in JVM *reader data-shape* differential test, not compilation.

Uses ONLY hardcoded small fixtures. JVM never sees source from user projects.
Default (no JARs) is a local baseline with exit code 3 (NOT CERTIFIED).
No downloads, no project execution, no runtime macro expansion, no CLASSPATH.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from analyzer.semantic_adapters.clojure_forms import ReaderError, lex, _Parser
from analyzer.semantic_adapters.clojure_shape import canonical_forms, UnsupportedShape
from atlas_clojure_reference import (REFERENCE_JARS, REFERENCE_VERSION,
                                     resolve_jar_dir, preflight_jars, sha)

SCHEMA = 'atlas-clojure-reader-structure/1'
# These forms are intentionally in a *restricted* canonical subset: no reader
# conditionals, syntax quote, auto-resolved namespaces, functions, JVM classes,
# regex/tag readers, macro expansion, or side-effecting project forms.
SHAPE_FIXTURES = (
    ('vector_scalars', '[true false nil :item example 5]'),
    ('simple_list', '(lookup source :key)'),
    ('simple_map', '{:a 1 :b 2}'),
    ('reordered_map', '{:b 2 :a 1}'),
    ('nested_collections', '[{:a 1} #{:x :y} (inc 2)]'),
    ('nested_map', '{:roles [:a :b] :count 2 :enabled true}'),
    ('nested_map_vector_keys', '{[1 2] {:ok true} [3 4] :other}'),
    ('map_set_key', '{#{:a :b} 1}'),
    ('set_atoms', '#{:alpha :beta symbol}'),
    ('set_nested_vectors', '#{[1 2] [3 4]}'),
    ('empty_collections', '[() [] {} #{}]'),
    ('int_and_bigint', '[1 1N -2 0]'),
    ('ratio_literal', '22/7'),
    ('ratio_simplified', '4/2'),
    ('namespaced_keyword', ':atlas/verified'),
    ('namespaced_symbol', 'atlas.reader/example'),
    ('escaped_string', r'"first\nsecond\u03A9"'),
    ('unicode_string', '"Příliš žluťoučký"'),
    ('character_named', r'[\newline \tab \space]'),
    ('character_unicode', r'\u03A9'),
    ('simple_control_form', '(if flag (f 1) (g 2))'),
    ('comments_and_commas', '[1, 2 ;comment\n 3]'.replace('\\n','\n')),
    ('discarded_form', '[1 #_ (unused) 2]'),
    ('discarded_map_pair', '{:a 1 #_ :hidden :b 2}'),
    ('keyword_map', '{:source/one {:target/other [1 2]}}'),
)

# Explicit negative fixtures test *acceptance*, NOT structural equivalence.
# These are separate from the 25 positive shape comparisons.
REJECTION_FIXTURES = (
    ('duplicate_keyword_map', '{:id 1 :id 2}'),
    ('duplicate_string_set', '#{"x" "x"}'),
    ('duplicate_bigint_map', '{1 :first 1N :second}'),
    ('duplicate_ratio_set', '#{2 4/2}'),
    ('duplicate_vector_set', '#{[1 2] [1 2]}'),
    ('duplicate_nested_map_key', '{{:a 1 :b 2} :one {:b 2 :a 1} :two}'),
)
TASKS = SHAPE_FIXTURES + REJECTION_FIXTURES


def _local_case(case_id: str, source: str) -> dict:
    try:
        forms = _Parser(lex(source, platform='clj')).all()
        canonical = canonical_forms(forms)
    except (ReaderError, ValueError, UnsupportedShape) as exc:
        return {'id':case_id, 'local_accepted':False, 'local_error':type(exc).__name__}
    return {'id':case_id,'local_accepted':True,'local_forms':len(forms),
            'local_shape_sha256':hashlib.sha256(canonical.encode('utf-8')).hexdigest()}


def local_baseline() -> list[dict]:
    return [dict(_local_case(case_id, source), expected_accepted=(case_id in
                  {name for name,_ in SHAPE_FIXTURES})) for case_id,source in TASKS]


def run_reference(jars: list[Path], java_exe: str='java', timeout: int=30, *, tasks=None, harness=None) -> tuple[str,dict]:
    tasks = TASKS if tasks is None else tuple(tasks)
    if len(tasks)>100 or len(tasks)!=len({case_id for case_id,_ in tasks}):
        raise ValueError('Invalid or duplicated reference fixture identifiers')
    java_path=shutil.which(java_exe)
    if java_path is None:
        raise ValueError('Java nenalezena')
    if len(jars)!=3:
        raise ValueError('Vyžaduji přesně tři lokální Clojure 1.12.6 JAR soubory')
    harness=Path(__file__).resolve().parent/'reference'/'atlas_clojure_safe_shape_reader.clj' if harness is None else Path(harness)
    payload=''.join(case_id+'\t'+base64.b64encode(source.encode('utf-8')).decode('ascii')+'\n'
                    for case_id,source in tasks)
    env={k:v for k,v in os.environ.items() if k in
         ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR','HOME','USERPROFILE')}
    with tempfile.TemporaryDirectory(prefix='atlas-clojure-shapes-') as work:
        cmd=[java_path,'-cp',os.pathsep.join(str(p.resolve()) for p in jars),
             'clojure.main',str(harness)]
        result=subprocess.run(cmd,input=payload,text=True,capture_output=True,
                              timeout=timeout,cwd=work,env=env,check=False)
    if result.returncode!=0:
        raise RuntimeError('Referenční JVM se nezdařila: '+result.stderr[-350:])
    version=None
    results={}
    for line in result.stdout.splitlines():
        parts=line.split('\t')
        if len(parts)==2 and parts[0]=='ATLAS_SHAPE_REF':
            if version is not None:raise RuntimeError('Duplicitní verze referenčního JVM')
            version=parts[1]
        elif len(parts)==5 and parts[0]=='ATLAS_SHAPE':
            _,case_id,status,n_text,raw=parts
            if case_id in results:raise RuntimeError('Duplicitní referenční strukturální výsledek')
            if status not in ('accepted','rejected'):raise RuntimeError('Neplatný referenční status')
            if not n_text.isdigit() or int(n_text)>1000:raise RuntimeError('Neplatný počet forem')
            count=int(n_text)
            if status=='accepted':
                if len(raw)>500_000:raise RuntimeError('Příliš velký referenční výstup')
                try:
                    shape=base64.b64decode(raw,validate=True).decode('utf8')
                except (ValueError,UnicodeError) as e:
                    raise RuntimeError('Neplatný base64 strukturální výstup') from e
                digest=hashlib.sha256(shape.encode('utf8')).hexdigest()
            else:
                if count!=0 or raw!='-':raise RuntimeError('Neplatná odpověď pro odmítnutou formu')
                digest=None
            results[case_id]={'reference_accepted': status=='accepted',
                              'reference_forms':count,
                              'reference_shape_sha256':digest}
        else:
            raise RuntimeError('Neznámý výstup referenčního JVM')
    if version!=REFERENCE_VERSION:
        raise RuntimeError(f'Očekáván Clojure {REFERENCE_VERSION}; JVM vrátila {version!r}')
    expected={id for id,_ in tasks}
    if set(results)!=expected:
        raise RuntimeError('JVM nevrátila přesnou množinu strukturálních testů')
    return version,results


def main(argv=None):
    parser=argparse.ArgumentParser(description='Bezpečné porovnání vybrané struktury Clojure forem bez spuštění projektu')
    parser.add_argument('--jar-dir',type=Path,help='Složka tří ručně ověřených Clojure JARů')
    parser.add_argument('--expected-clojure-sha256',help='SHA-256 z nezávislého důvěryhodného zdroje')
    parser.add_argument('--check-jars',action='store_true',help='Pouze předkontrola JARů, bez spuštění JVM')
    parser.add_argument('--java',default='java')
    parser.add_argument('--timeout',type=int,default=30)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.timeout<1 or args.timeout>180:parser.error('--timeout 1 až 180 sekund')
    cases=local_baseline()
    report={
        'schema':SCHEMA,
        'reference_release':REFERENCE_VERSION,
        'scope':'Selected Clojure reader DATA-SHAPE subset only; not complete AST, macros, bindings, CFG/DFG or code execution',
        'fixture_corpus_sha256':hashlib.sha256(json.dumps(TASKS,ensure_ascii=False).encode('utf8')).hexdigest(),
        'jvm_harness_sha256':sha(Path(__file__).resolve().parent/'reference'/'atlas_clojure_safe_shape_reader.clj'),
        'case_count':len(cases),
        'positive_shape_case_count':len(SHAPE_FIXTURES),
        'negative_reader_case_count':len(REJECTION_FIXTURES),
        'local_shape_baseline_passed':sum(x['local_accepted'] for x in cases if x['expected_accepted']),
        'local_negative_baseline_passed':sum(not x['local_accepted'] for x in cases if not x['expected_accepted']),
        'reference_runtime_executed':False,'reference_binary_origin_attested':False,
        'source_or_project_code_executed':False,'network_requested':False,
        'forensic_or_semantic_certification':False,'full_reader_or_ast_conformance':False,
        'cases':cases,
    }
    if args.expected_clojure_sha256:
        if not re.fullmatch('[0-9A-Fa-f]{64}',args.expected_clojure_sha256):
            parser.error('SHA-256 musí mít 64 hex znaků')
        if not args.jar_dir:parser.error('--expected-clojure-sha256 vyžaduje --jar-dir')
    if args.check_jars and not args.jar_dir:parser.error('--check-jars vyžaduje --jar-dir')
    if args.jar_dir:
        jars=resolve_jar_dir(args.jar_dir)
        checked=preflight_jars(jars)
        report['preflight_passed']=True
        # Do not export a user's machine paths in the shareable report.
        report['preflight_jars']=[{'name':path.name,'sha256':row['sha256'],
                                   'bytes':row['bytes'], 'publisher_signature_verified':False}
                                  for path,row in zip(jars,checked)]
        if args.expected_clojure_sha256 and sha(jars[0]).lower()!=args.expected_clojure_sha256.lower():
            raise ValueError('JAR SHA-256 nesouhlasí: referenční JVM NEBYLA spuštěna')
        report['user_supplied_digest_matched']=bool(args.expected_clojure_sha256)
        if not args.check_jars:
            version,reference=run_reference(jars,args.java,args.timeout)
            report['reference_runtime_executed']=True
            report['runtime_version']=version
            for row in cases:
                ref=reference[row['id']]
                row.update(ref)
                if row['expected_accepted']:
                    row['matches_reference']=(row['local_accepted'] and ref['reference_accepted'] and
                         row['local_forms']==ref['reference_forms'] and
                         row['local_shape_sha256']==ref['reference_shape_sha256'])
                else:
                    row['matches_reference']=(not row['local_accepted'] and
                                               not ref['reference_accepted'] and
                                               ref['reference_forms']==0)
            report['matches_reference']=sum(x['matches_reference'] for x in cases)
            report['positive_shapes_matching_reference']=sum(x['matches_reference'] for x in cases
                                                          if x['expected_accepted'])
            report['negative_cases_matching_reference']=sum(x['matches_reference'] for x in cases
                                                         if not x['expected_accepted'])
            report['structure_subset_conformance_passed']=(report['matches_reference']==len(cases))
    if not report['reference_runtime_executed']:
        report['structure_subset_conformance_passed']=False
        report['reference_not_run_reason']='No JARs or preflight only: actual JVM comparison NOT run.'
    if args.output:
        if args.output.is_symlink():raise ValueError('Výstupní cesta nesmí být symbolický odkaz')
        target=args.output.resolve()
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Atlas shape baseline: {report["local_shape_baseline_passed"]}/{len(SHAPE_FIXTURES)}')
    print(f'Atlas negative reader baseline: {report["local_negative_baseline_passed"]}/{len(REJECTION_FIXTURES)}')
    if report['reference_runtime_executed']:
        print(f'Clojure JVM {report["runtime_version"]} shapes: '
              f'{report["positive_shapes_matching_reference"]}/{len(SHAPE_FIXTURES)}; '
              f'negative cases: {report["negative_cases_matching_reference"]}/{len(REJECTION_FIXTURES)}')
    else:
        print('REFERENCE JVM NEBYLA SPUŠTĚNA. Shoda struktur není potvrzena.')
    if args.output:print('Report:',args.output)
    if not report['reference_runtime_executed']:return 3
    return 0 if report['structure_subset_conformance_passed'] else 2

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as e:
        print('CHYBA REFERENČNÍ STRUKTURY:',str(e),file=sys.stderr)
        print('Uživatelský projekt nebyl analyzován ani spuštěn.',file=sys.stderr)
        sys.exit(4)
