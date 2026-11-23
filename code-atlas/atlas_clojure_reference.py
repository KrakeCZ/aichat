#!/usr/bin/env python3
"""Opt-in local Clojure JVM READER reference check on built-in, trusted fixtures.

No project files, project code, REPL, server, build tool, network or dependency
resolution. ONLY explicit --clojure-jar runs a JVM; default is Python-only.
The goal is to detect syntax acceptance mismatches, not to certify that Atlas
correctly resolves calls, macro expansion, dynamic code or all reader forms.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
import re

from analyzer.semantic_adapters.clojure_forms import lex, _Parser, ReaderError

REFERENCE_VERSION='1.12.6'
# Exact pinned upstream runtime dependency filenames. This controls only the
# optional reference-fixture JVM, never the analyzed project's dependencies.
REFERENCE_JARS = (
    ('clojure', 'clojure-1.12.6.jar'),
    ('spec.alpha', 'spec.alpha-0.5.238.jar'),
    ('core.specs.alpha', 'core.specs.alpha-0.4.74.jar'),
)
JAR_PRECHECK_MAX_BYTES = 80_000_000
JAR_PRECHECK_MAX_ENTRIES = 100_000


def resolve_jar_dir(folder: Path) -> list[Path]:
    """Resolve just three explicit filenames in the chosen local directory.

    Never recurse into a home directory, Maven cache, analyzed repository or
    CLASSPATH. Never download or invoke any tool.
    """
    folder = folder.expanduser()
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(
            f'Zadana slozka s Clojure JARy neexistuje: {folder}\n'
            'Vytvor ji a nakopiruj do ni tri rucne overene JAR soubory. '
            'Nemusis pouzivat C:\\tools\\clojure; to byla jen ukazkova cesta.')
    return [folder / filename for _, filename in REFERENCE_JARS]


def preflight_jars(jars: list[Path]) -> list[dict]:
    """Inspect local ZIP container metadata. Never load or execute any JAR.

    Checks format and plausibility, NOT upstream publisher identity/signature.
    No validation here can make an untrusted JVM artifact safe to execute.
    """
    result=[]
    for index, jar in enumerate(jars):
        jar=jar.expanduser()
        if jar.is_symlink() or not jar.is_file():
            raise ValueError(
                f'Expected a regular, non-symlinked local JAR: {jar}\n'
                'Chybi pozadovany soubor. '
                'Puvodni C:\\tools\\clojure byla pouze prikladova cesta. '
                'Oficialni verze: https://clojure.org/releases/downloads')
        if jar.suffix.lower() != '.jar':
            raise ValueError(f'Soubor nema priponu .jar: {jar}')
        size=jar.stat().st_size
        if not 0 < size <= JAR_PRECHECK_MAX_BYTES:
            raise ValueError(f'JAR ma neocekavanou velikost: {jar}')
        try:
            with zipfile.ZipFile(jar,'r') as f:
                entries=f.infolist()
                if not entries or len(entries)>JAR_PRECHECK_MAX_ENTRIES:
                    raise ValueError(f'JAR nema validni obsah: {jar}')
                names={i.filename for i in entries}
                if index == 0 and 'clojure/main.class' not in names:
                    raise ValueError(
                        f'JAR neobsahuje clojure/main.class: {jar}. '
                        'Over, ze jde skutecne o oficiální Clojure runtime 1.12.6.')
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            raise ValueError(f'JAR neni platny ZIP archiv: {jar}') from exc
        result.append({'path':str(jar.resolve()),'bytes':size,'sha256':sha(jar),
                       'archive_structure_checked': True,
                       'publisher_signature_verified': False})
    return result
# Only these static trusted fixtures can be sent to the optional JVM process.
# They must never be derived from an analyzed project or CLI-supplied source.
FIXTURES = (
    ('bare_symbolic_inf', 'clj', '##Inf (defn f [] 1)', True, 2),
    ('bare_symbolic_minus_inf', 'clj', '##-Inf (defn f [] 1)', True, 2),
    ('bare_symbolic_nan', 'clj', '##NaN (defn f [] 1)', True, 2),
    ('invalid_symbolic_value', 'clj', '##NoSuchValue', False, 0),
    ('one_char', 'clj', r'(ns a) \x', True, 2),
    ('named_character', 'clj', r'(ns a) \newline', True, 2),
    ('unicode_character', 'clj', r'(ns a) \u03A9', True, 2),
    ('octal_character', 'clj', r'(ns a) \o377', True, 2),
    ('invalid_named_character', 'clj', r'(ns a) \unknown', False, 0),
    ('invalid_short_unicode', 'clj', r'(ns a) \u123', False, 0),
    ('invalid_unicode_surrogate', 'clj', r'(ns a) \uD800', False, 0),
    ('invalid_octal_range', 'clj', r'(ns a) \o400', False, 0),
    ('invalid_octal_length', 'clj', r'(ns a) \o1234', False, 0),
    ('valid_map_pairs', 'clj', '{:a 1 :b 2}', True, 1),
    ('invalid_odd_map', 'clj', '{:a 1 :b}', False, 0),
    ('discarded_map_entries', 'clj', '{:a 1 #_ :hidden :b 2}', True, 1),
    ('unterminated_map', 'clj', '{:a 1', False, 0),
    ('nested_collections', 'clj', '[{:a 1} #{:x :y} (inc 2)]', True, 1),
    ('quoted_expression', 'clj', "'(ns no.execution)", True, 1),
    ('read_eval_denied', 'clj', '#=(System/exit 1)', False, 0),
    ('valid_string_escape', 'clj', r'"a\n\u03A9"', True, 1),
    ('invalid_string_escape', 'clj', r'"a\q"', False, 0),
    ('regex_not_compiled_here', 'clj', r'#"\w+"', True, 1),
    ('shared_reader_conditional', 'cljc', '#?(:clj 1 :cljs 2)', True, 1),
    ('shared_reader_cond_splice', 'cljc', '[1 #?@(:clj [2 3] :cljs [4])]', True, 1),
    ('invalid_top_level_splice', 'cljc', '#?@(:clj [1])', False, 0),
    ('namespaced_map', 'clj', '#:inventory{:id 1}', True, 1),
    ('tagged_uuid', 'clj', '#uuid "3b8a31ed-fd89-4f1b-a00f-42e3d60cf5ce"', True, 1),
    # Additional fixed reference cases derived from the official Reader spec.
    # They test syntax acceptance/form counts only, never full semantic AST.
    ('reader_quoted_data', 'clj', "'(fn [] (danger))", True, 1),
    ('reader_var_quote', 'clj', "#'clojure.core/map", True, 1),
    ('reader_anonymous_fn', 'clj', '#(+ % 1)', True, 1),
    ('reader_discard', 'clj', '#_ (unseen) (ns kept)', True, 1),
    ('reader_auto_keyword', 'clj', '::local', True, 1),
    ('reader_valid_set', 'clj', '#{1 2 3}', True, 1),
    ('reader_ratio_number', 'clj', '22/7', True, 1),
    ('reader_commas', 'clj', '[1, 2, 3]', True, 1),
    ('reader_line_comment', 'clj', '(ns example);ignored\n(def x true)', True, 2),
    ('invalid_close_delimiter', 'clj', ')', False, 0),
    ('invalid_unbalanced_vector', 'clj', '[1 2', False, 0),
    ('invalid_unterminated_regex', 'clj', '#"unclosed', False, 0),
)


def baseline():
    cases=[]
    for case_id, platform, source, expected_ok, expected_count in FIXTURES:
        try:
            count=len(_Parser(lex(source, platform=('shared' if platform=='cljc' else platform))).all())
            ok=True
        except (ReaderError, ValueError):
            count=0
            ok=False
        cases.append({'id':case_id,'platform':platform,'expected_accepted':expected_ok,
                      'expected_forms':expected_count,'atlas_accepted':ok,'atlas_forms':count,
                      'matches_expected':(ok==expected_ok and (not ok or count==expected_count))})
    return cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_reference(jars: list[Path], java_exe: str, timeout: int, *, fixtures=None):
    """Run source-mode (allow) and auxiliary preservation-mode probes.

    For .cljc, :allow is the source-loading check. :preserve retains
    conditional forms as data for static tooling and is NOT equivalent to
    source acceptance. For .clj, reader conditionals remain disabled.

    Returns (version, source_results, auxiliary_preserve_results).
    """
    fixtures=FIXTURES if fixtures is None else tuple(fixtures)
    if not fixtures or len(fixtures)>100 or len({f[0] for f in fixtures})!=len(fixtures):
        raise ValueError('invalid fixture identifiers/count')
    java_path=shutil.which(java_exe)
    if not java_path:
        raise ValueError(f'Java executable not found: {java_exe}')
    if not jars:
        raise ValueError('Clojure official JAR path is required')
    for jar in jars:
        if jar.is_symlink() or not jar.is_file() or jar.suffix.lower() != '.jar':
            raise ValueError(f'Expected a regular, non-symlinked local JAR: {jar}')

    # Primary :allow mode for .cljc; ordinary read for .clj.
    # Auxiliary :preserve mode probes ONLY the .cljc examples. None of these
    # payloads come from source projects or user-supplied arguments.
    tasks=[(case_id,platform,'allow',source)
           for case_id,platform,source,_,_ in fixtures]
    tasks += [(case_id,platform,'preserve',source)
              for case_id,platform,source,_,_ in fixtures if platform=='cljc']
    payload=''.join(f'{case_id}\t{platform}\t{mode}\t{base64.b64encode(code.encode("utf-8")).decode("ascii")}\n'
                    for case_id,platform,mode,code in tasks)
    harness=Path(__file__).resolve().parent/'reference'/'atlas_clojure_safe_fixture_reader.clj'
    env={k:v for k,v in os.environ.items() if k in
         ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR','HOME','USERPROFILE')}
    with tempfile.TemporaryDirectory(prefix='atlas-clojure-reference-') as work:
        cmd=[java_path,'-cp',os.pathsep.join(str(p.resolve()) for p in jars),
             'clojure.main',str(harness)]
        result=subprocess.run(cmd,input=payload,text=True,capture_output=True,
                              timeout=timeout,cwd=work,env=env,check=False)
    if result.returncode != 0:
        raise RuntimeError('Trusted Clojure JVM reference failed to launch or read fixtures. '
                           'Verify official JARs and their dependencies; stderr excerpt: '
                           +result.stderr[-450:])
    version=None
    results={}
    for line in result.stdout.splitlines():
        parts=line.split('\t')
        if len(parts)==2 and parts[0]=='ATLAS_REF':
            version=parts[1]
        if len(parts)==5 and parts[0]=='ATLAS_CASE':
            mode,case_id,status,count=parts[1:]
            if mode not in {'allow','preserve'} or status not in {'accepted','rejected'}:
                raise RuntimeError('Unexpected reference probe mode or status')
            key=(mode,case_id)
            if key in results:
                raise RuntimeError('Duplicate reference probe result')
            try:
                n=int(count)
            except ValueError:
                raise RuntimeError('Invalid reference case count')
            if n < 0:
                raise RuntimeError('Negative reference case count')
            results[key]={'accepted':status=='accepted','forms':n}
    if version != REFERENCE_VERSION:
        raise RuntimeError(f'Expected Clojure {REFERENCE_VERSION}, got {version!r}. '
                           'Comparison is NOT validated for a different version.')
    required={(mode,case_id) for case_id,_,mode,_ in tasks}
    if set(results)!=required:
        missing=sorted(required-set(results))[:4]
        raise RuntimeError(f'Clojure JVM did not return the exact expected fixture modes: {missing}')
    primary={name:results['allow',name] for name,_,_,_,_ in fixtures}
    preserved={name:results['preserve',name]
               for name,platform,_,_,_ in fixtures if platform=='cljc'}
    return version,primary,preserved


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline Clojure reader conformance; never executes an analyzed project')
    parser.add_argument('--jar-dir',type=Path,help='Folder containing all three official pinned JARs; explicitly opt in')
    parser.add_argument('--check-jars',action='store_true',help='Check files only; NEVER start Java; exit code 3 even when ready')
    parser.add_argument('--clojure-jar',type=Path,help='LOCAL official org.clojure/clojure 1.12.6 jar; explicit opt-in')
    parser.add_argument('--spec-alpha-jar',type=Path,help='Optional local Clojure spec.alpha dependency')
    parser.add_argument('--core-specs-alpha-jar',type=Path,help='Optional local core.specs.alpha dependency')
    parser.add_argument('--expected-clojure-sha256',help='Independent expected SHA-256 of your trusted Clojure JAR')
    parser.add_argument('--java',default='java',help='Local Java executable')
    parser.add_argument('--timeout',type=int,default=30)
    parser.add_argument('--output',type=Path,help='Where to save the local JSON result')
    args=parser.parse_args(argv)
    if args.jar_dir and any(p is not None for p in (
        args.clojure_jar,args.spec_alpha_jar,args.core_specs_alpha_jar)):
        parser.error('--jar-dir nelze kombinovat s jednotlivymi --*-jar cestami')
    if args.jar_dir:
        jars=resolve_jar_dir(args.jar_dir)
        args.clojure_jar,args.spec_alpha_jar,args.core_specs_alpha_jar=jars
    elif args.check_jars and not args.clojure_jar:
        parser.error('--check-jars vyzaduje --jar-dir nebo --clojure-jar')
    cases=baseline()
    corpus_bytes=json.dumps(FIXTURES,ensure_ascii=False,sort_keys=True).encode('utf-8')
    report={'schema':'atlas-clojure-reference-acceptance/1',
            'fixture_corpus_sha256':hashlib.sha256(corpus_bytes).hexdigest(),
            'reference_harness_sha256':sha(Path(__file__).resolve().parent/'reference'/'atlas_clojure_safe_fixture_reader.clj'),
            'reference_release':REFERENCE_VERSION,
            'source_reference_modes': {'clj': 'ordinary_read', 'cljc': ':read-cond :allow'},
            'preservation_mode_is_not_source_acceptance': True,
            'auxiliary_preserve_mode': ':read-cond :preserve (cljc only)',

            'reference_runtime_executed':False,
            'reference_binary_origin_attested':False,
            'user_supplied_digest_matched':False,
            'project_code_executed':False,
            'network_requested':False,
            'semantic_or_forensic_certification':False,
            'scope':'Fixed trusted syntax-acceptance fixtures only; NOT AST equivalence, symbol resolution or runtime behavior',
            'reference_protocol_schema':2,
            'full_language_conformance_passed':False,
            'source_compilation_or_macroexpansion_executed':False,
            'case_count':len(cases),
            'local_baseline_passed':sum(c['matches_expected'] for c in cases),
            'cases':cases}
    if args.expected_clojure_sha256 and not args.clojure_jar:
        parser.error('--expected-clojure-sha256 requires --clojure-jar')
    if args.expected_clojure_sha256 and not re.fullmatch('[0-9a-fA-F]{64}',args.expected_clojure_sha256):
        parser.error('--expected-clojure-sha256 musi byt 64 hex znaku')
    if args.clojure_jar:
        jars=[args.clojure_jar]+[p for p in (args.spec_alpha_jar,args.core_specs_alpha_jar) if p is not None]
        if args.expected_clojure_sha256:
            actual=sha(args.clojure_jar)
            if actual.lower()!=args.expected_clojure_sha256.lower():
                raise SystemExit('Clojure JAR SHA-256 mismatch; did not execute runtime')
            report['user_supplied_digest_matched']=True  # does NOT attest the publisher
        else:
            report['user_supplied_digest_matched']=False
        checked=preflight_jars(jars)
        report['preflight_jar_files']=checked
        report['preflight_passed']=True
        if args.check_jars:
            report['differential_conformance_passed']=False
            report['reference_not_run_reason']='Preflight-only: JVM deliberately not executed.'
            if args.output:
                target=args.output.resolve()
                if args.output.is_symlink():
                    raise ValueError(f'Vystupni cesta je symlink: {args.output}')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print('PREDKONTROLA JARU USPECH: soubory existuji a maji citelnou strukturu.')
            for row in checked:
                print(' ',row['path'],'SHA-256:',row['sha256'])
            print('REFERENCNI TEST NESPUSTEN. Pro porovnani spust stejny prikaz bez --check-jars.')
            print('Pozor: struktura ZIP neni overeni pravosti vydavatele.')
            return 3  # Never signal completed conformance without running JVM.
        version, ref_cases,preserve_cases=run_reference(jars,args.java,args.timeout)
        report['reference_runtime_executed']=True
        report['runtime_version']=version
        report['preserve_mode_probes']=[
            {'id':name,'mode':':read-cond :preserve',
             'reference_accepted':preserve_cases[name]['accepted'],
             'reference_forms':preserve_cases[name]['forms'],
             'used_for_source_conformance':False}
            for name in sorted(preserve_cases)]
        report['reference_jar_sha256']=sha(args.clojure_jar)
        for case in cases:
            reference=ref_cases[case['id']]
            case['reference_read_mode']=(':read-cond :allow' if case['platform']=='cljc'
                                         else 'ordinary_read')
            case['reference_accepted']=reference['accepted']
            case['reference_forms']=reference['forms']
            case['matches_reference']=(case['atlas_accepted']==reference['accepted'] and
                    (not case['atlas_accepted'] or case['atlas_forms']==reference['forms']))
        report['matches_reference']=sum(c['matches_reference'] for c in cases)
        report['differential_conformance_passed']=(report['matches_reference']==len(cases))
    else:
        report['differential_conformance_passed']=False
        report['reference_not_run_reason']='Official Clojure 1.12.6 JAR not provided. Java alone is insufficient.'
    if args.output:
        if args.output.is_symlink():
            raise ValueError(f'Vystupni cesta je symlink: {args.output}')
        target=args.output.resolve()
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Atlas reader fixtures: {report["local_baseline_passed"]}/{len(cases)} local baseline')
    if args.clojure_jar:
        print(f'Clojure {report["runtime_version"]} comparison: {report["matches_reference"]}/{len(cases)} syntax cases')
        if not report['reference_binary_origin_attested']:
            print('REFERENCE BINARY ORIGIN NOT ATTESTED: verify JAR against publisher signature/hash.')
    else:
        print('REFERENCE NOT EXECUTED: no Clojure JAR supplied. Full language conformance NOT proven.')
    if args.output:
        print('Report:',args.output.resolve())
    # Fail closed for CI: passing local synthetic examples without the official
    # reader is NOT a successful reference conformance check.
    if not args.clojure_jar:
        return 3
    return 0 if report['differential_conformance_passed'] else 2

if __name__=='__main__':
    # End-user interface must not dump a technical Python traceback for a
    # missing sample path. Keep direct main() exceptions testable by pytest.
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError,
            OSError, subprocess.TimeoutExpired) as exc:
        print('CHYBA REFERENCNI KONTROLY:',str(exc),file=sys.stderr)
        print('Jak pokracovat: vloz skutecne oficialni soubory do jedne slozky',file=sys.stderr)
        print('  clojure-1.12.6.jar, spec.alpha-0.5.238.jar, core.specs.alpha-0.4.74.jar',file=sys.stderr)
        print('  py -3 atlas_clojure_reference.py --jar-dir "C:\\vlastni\\slozka" --check-jars',file=sys.stderr)
        print('Zdroj: https://clojure.org/releases/downloads',file=sys.stderr)
        print('Java ANI REFERENCNI KOMPARACE NEBYLY USPESNE DOKONCENY.',file=sys.stderr)
        sys.exit(4)
