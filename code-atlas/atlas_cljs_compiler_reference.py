#!/usr/bin/env python3
"""EXPLICIT opt-in, local ClojureScript COMPILER oracle on Atlas-owned fixtures.

This harness NEVER compiles an analyzed user's project. It creates a fixed,
short fixture corpus in isolated temp directories and invokes only an
explicitly hash-approved local standalone ClojureScript JAR. No downloader,
Maven, clj, gradle, npm, shell or arbitrary fixture paths are accepted.

IMPORTANT: Compilation executes *trusted, bundled fixture macros*, and JVM
bytecode from the manually supplied JAR. An SHA-256 match is not an upstream
signature. Run only if you trust the exact local binary. Without --run and a
valid JAR digest, reference conformance MUST be reported as NOT EXECUTED.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from analyzer.semantic_adapters.clojure_forms import analyze_clojure

SCHEMA = 'atlas-cljs-macro-compiler-oracle/1'
DOCUMENTED_CLJS_VERSION = '1.12.145'
REQUIRED_CLASSES = frozenset({'cljs/main.class', 'clojure/lang/RT.class'})
MAX_JAR_BYTES = 180_000_000
MAX_COMPILER_LOG = 4000
MAX_VERSION_METADATA_BYTES = 32_768

BASE_MACROS = '(ns atlas.oracle.macros)\n(defmacro add-one [x] (list (symbol "+") x 1))\n'
RUNTIME_LIB = '(ns atlas.oracle.macros)\n(defn plain [] 2)\n'


@dataclass(frozen=True)
class Fixture:
    key: str
    consumer: str
    provider: str = BASE_MACROS
    provider_suffix: str = '.clj'
    runtime_lib: str | None = None
    expected_decl_status: str | None = 'single_public_direct_defmacro_declaration_candidate'
    required_refer: tuple[str, ...] = ('add-one',)
    compiler_expectation: str = 'compile_success'
    origin: str = ':require-macros'
    expected_js_marker: str | None = None
    expected_candidate_count: int = 1


FIXTURES = (
    Fixture('explicit_refer',
        '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [add-one]]))\n'
        '(defn entry [] (add-one 2))\n'),
    Fixture('explicit_alias',
        '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :as m]))\n'
        '(defn entry [] (m/add-one 2))\n',required_refer=(),expected_decl_status=None),
    Fixture('inline_refer',
        '(ns atlas.oracle.client (:require [atlas.oracle.macros :refer [plain] :refer-macros [add-one]]))\n'
        '(defn entry [] (add-one (plain)))\n',runtime_lib=RUNTIME_LIB,
        origin=':require/:refer-macros'),
    Fixture('inline_include',
        '(ns atlas.oracle.client (:require [atlas.oracle.macros :as m :include-macros true]))\n'
        '(defn entry [] (m/add-one (m/plain)))\n',runtime_lib=RUNTIME_LIB,
        required_refer=(),expected_decl_status=None,origin=':require/:include-macros'),
    Fixture('use_macros_only',
        '(ns atlas.oracle.client (:use-macros [atlas.oracle.macros :only [add-one]]))\n'
        '(defn entry [] (add-one 2))\n',origin=':use-macros/:only'),
    Fixture('cljc_macro_provider',
        '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [add-one]]))\n'
        '(defn entry [] (add-one 2))\n',provider_suffix='.cljc'),
    # Negative controls record compiler behavior without asserting that the
    # compiler MUST reject a source solely based on our structural evidence.
    Fixture('missing_direct_macro_observation',
        '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [missing]]))\n'
        '(defn entry [] (missing 2))\n',
        expected_decl_status='no_literal_defmacro_found_not_proof_of_absence',
        required_refer=('missing',),compiler_expectation='observe_only'),
    Fixture('private_macro_observation',
        '(ns atlas.oracle.client (:require-macros [atlas.oracle.macros :refer [secret]]))\n'
        '(defn entry [] (secret 2))\n',
        provider='(ns atlas.oracle.macros)\n(defmacro ^:private secret [x] (list (symbol "+") x 1))\n',
        expected_decl_status='private_direct_defmacro_not_public_candidate',
        required_refer=('secret',),compiler_expectation='observe_only'),
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sources(fixture: Fixture) -> dict[str,str]:
    out = {
        'atlas/oracle/client.cljs':fixture.consumer,
        'atlas/oracle/macros'+fixture.provider_suffix:fixture.provider,
    }
    if fixture.runtime_lib is not None:
        out['atlas/oracle/macros.cljs']=fixture.runtime_lib
    return out


def _write_sources(root: Path, fixture: Fixture) -> dict[str,str]:
    files=_sources(fixture)
    for name,source in files.items():
        target=root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(source,encoding='utf-8')
    return {name:_sha(source.encode('utf-8')) for name,source in sorted(files.items())}


def _local_evidence(fixture: Fixture, sources_root: Path) -> dict:
    files=_sources(fixture)
    graph,warnings=analyze_clojure(sources_root,
        {p:{'language':'clojure'} for p in files},{})
    output={'atlas_candidate_confirmed':False,'warnings':warnings,
            'candidate_count':None,'direct_declaration_status':None,
            'macro_symbols_verified':False,'source_graph_hash':None}
    consumer=graph.get('atlas/oracle/client.cljs')
    if not consumer:
        output['error']='consumer_absent_or_unreadable'
        return output
    output['source_graph_hash']=consumer.get('hash')
    phase=consumer.get('platform_macro_phase_evidence',{})
    rows=phase.get('candidates',[])
    output['candidate_count']=len(rows)
    if len(rows)==0 and fixture.expected_candidate_count==0:
        output['atlas_candidate_confirmed']=not warnings
        return output
    if len(rows)!=1 or fixture.expected_candidate_count!=1:
        output['error']='not_one_unique_macro_provider_candidate'
        return output
    row=rows[0]
    if row.get('macro_symbols_verified') or row.get('semantic_resolution_certified'):
        output['error']='candidate_was_promoted_to_semantically_verified'
        return output
    if row.get('origin') != fixture.origin or row.get('namespace')!='atlas.oracle.macros':
        output['error']='macro_origin_or_namespace_mismatch'
        return output
    direct=row.get('direct_macro_declaration_evidence',[])
    if tuple(row.get('requested_refer_macros',()))!=fixture.required_refer:
        output['error']='requested_macro_names_mismatch'
        return output
    if len(direct)==1:
        output['direct_declaration_status']=direct[0].get('status')
    if fixture.expected_decl_status is None:
        good = len(direct)==0
    else:
        good = (len(direct)==1 and direct[0]['status']==fixture.expected_decl_status)
    output['atlas_candidate_confirmed']=bool(good and not warnings)
    if not good:
        output['error']='direct_defmacro_symbol_evidence_mismatch'
    return output


def _jar_version_evidence(z: zipfile.ZipFile) -> dict:
    """Read two fixed, small embedded source/metadata files; no JVM execution.

    A standalone build can contain stale, generated or in-development version
    strings. Neither version string is a publisher-attested release identity.
    """
    pom='META-INF/maven/org.clojure/clojurescript/pom.properties'
    core='cljs/core.cljs'
    out={'pom_version':None,'core_clojurescript_version':None,
         'documented_latest_stable_target':DOCUMENTED_CLJS_VERSION,
         'official_release_digest_verified':False,
         'embedded_version_metadata_conflict':False}
    def small_text(name: str) -> str | None:
        try:
            meta=z.getinfo(name)
        except KeyError:
            return None
        # The compiler core.cljs is a large source file. Read only a bounded
        # prefix: its generated version declaration is near the beginning.
        # This avoids loading large/untrusted JAR members into memory.
        with z.open(meta) as member:
            prefix=member.read(MAX_VERSION_METADATA_BYTES)
        if meta.file_size>MAX_VERSION_METADATA_BYTES:
            out['version_metadata_prefix_only']=True
        return prefix.decode('utf8','replace')
    data=small_text(pom)
    if data is not None:
        match=re.search(r'(?m)^version=([^\s]+)\s*$',data)
        if match:out['pom_version']=match.group(1)
    data=small_text(core)
    if data is not None:
        match=re.search(r'\(def\s+\*clojurescript-version\*\s+"([^"]+)"\)',data)
        if match:out['core_clojurescript_version']=match.group(1)
    p=out['pom_version'];c=out['core_clojurescript_version']
    out['embedded_version_metadata_conflict']=bool(p and c and p!=c)
    out['version_identity_status']=('conflicting_embedded_versions_not_certified' if p and c and p!=c
                                    else 'internally_consistent_not_publisher_attested' if p and c
                                    else 'partial_version_metadata_not_certified' if p or c
                                    else 'version_metadata_missing_not_certified')
    return out


def _jar_preflight(path: Path | None, digest: str | None) -> tuple[dict,Path | None]:
    evidence={'status':'not_run_missing_or_unverified_jar',
              'standalone_jar_structure_checked':False,
              'sha256_matched_user_supplied_expected':False,
              'publisher_signature_verified':False}
    if path is None:
        evidence['reason']='standalone_cljs_jar_not_provided'
        return evidence,None
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_JAR_BYTES:
        evidence['reason']='missing_symlink_or_oversized_jar'
        return evidence,None
    raw_hash=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            raw_hash.update(block)
    actual=raw_hash.hexdigest()
    evidence.update({'jar_sha256':actual,'jar_size_bytes':path.stat().st_size})
    if not digest or len(digest)!=64 or any(ch not in '0123456789abcdefABCDEF' for ch in digest):
        evidence['reason']='expected_sha256_required_for_opt_in'
        return evidence,None
    if actual.lower()!=digest.lower():
        evidence['reason']='jar_sha256_mismatch'
        return evidence,None
    try:
        with zipfile.ZipFile(path) as z:
            classes=set(z.namelist())
            if not REQUIRED_CLASSES.issubset(classes):
                evidence['reason']='not_a_standalone_cljs_main_jar'
                return evidence,None
            # Read only the two fixed small class members for a bounded CRC
            # smoke check; all other bytecode is intentionally untrusted.
            for name in REQUIRED_CLASSES:
                meta=z.getinfo(name)
                if meta.file_size>1_000_000:
                    evidence['reason']='suspicious_class_size'
                    return evidence,None
                z.read(name)
            # Report all observed version identifiers, do not choose between
            # contradictory sources and do not claim release provenance.
            evidence['version_evidence']=_jar_version_evidence(z)
    except (OSError,ValueError,zipfile.BadZipFile,RuntimeError) as exc:
        evidence['reason']='invalid_jar_archive_'+type(exc).__name__
        return evidence,None
    evidence.update({'status':'passed_local_preflight',
                     'standalone_jar_structure_checked':True,
                     'sha256_matched_user_supplied_expected':True})
    evidence.pop('reason',None)
    return evidence,path.resolve()


def _compiler_result(jar: Path, java: str, fixture: Fixture, root: Path,
                     timeout: int) -> dict:
    src=root/'src'; out=root/'output'
    out.mkdir()
    # Fixed namespace, fixed optimizations, no REPL, watch, npm, tools.deps,
    # arbitrary -e, project configuration or script. No shell=True.
    cmd=[java,'-Xmx768m','-Djava.awt.headless=true',
         '-Duser.home='+str(root),'-Djava.io.tmpdir='+str(root),
         '-cp',str(jar)+os.pathsep+str(src),
         'cljs.main','-d',str(out),'-O','none','-c','atlas.oracle.client']
    # Inherit only essentials for Java executable access. Explicitly remove
    # CLASSPATH and JAVA_TOOL_OPTIONS to avoid silently added dependencies.
    allowed={'PATH','SYSTEMROOT','SystemRoot','WINDIR','TMP','TEMP','TMPDIR',
             'HOME','USERPROFILE','LANG','LC_ALL'}
    env={k:v for k,v in os.environ.items() if k in allowed}
    try:
        completed=subprocess.run(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,
                                 capture_output=True,text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        return {'status':'compiler_timeout','compiled':False,'timed_out':True,
                'timeout_seconds':timeout}
    except (OSError,ValueError) as exc:
        return {'status':'compiler_process_error','compiled':False,
                'error_type':type(exc).__name__}
    compiled=(out/'atlas/oracle/client.js').is_file()
    marker_found=None
    if fixture.expected_js_marker is not None:
        marker_found=(compiled and fixture.expected_js_marker in
                      (out/'atlas/oracle/client.js').read_text(encoding='utf-8'))
    ok=completed.returncode==0 and compiled and marker_found is not False
    # Logs are from trusted synthetic fixtures but may include host paths;
    # keep only short tails, without paths to any analyzed user project.
    stdout=completed.stdout[-MAX_COMPILER_LOG:]
    stderr=completed.stderr[-MAX_COMPILER_LOG:]
    # A JVM/bootstrap failure says nothing about acceptance of the fixture.
    # Inspect the complete captured diagnostic before truncating the report.
    environment_failure=completed.returncode!=0 and any(marker in completed.stderr+completed.stdout
        for marker in ('UnsupportedClassVersionError', 'NoClassDefFoundError',
                       'ClassNotFoundException', 'UnsatisfiedLinkError',
                       'Error occurred during initialization of VM',
                       'error while loading shared libraries', 'OutOfMemoryError'))
    return {'status':'compiler_compiled' if ok else
                    ('compiler_environment_error' if environment_failure else
                     'compiler_exit_zero_without_client_js' if completed.returncode==0
                      else 'compiler_failed'),
            'compiled':ok,'compiler_exit_code':completed.returncode,
            'output_client_js_exists':compiled,
            'expansion_marker_found':marker_found,
            'stderr_head':completed.stderr[:MAX_COMPILER_LOG],
            'stderr_tail':stderr,'stdout_tail':stdout}


def run_reference(*, jar: Path | None = None, expected_sha256: str | None = None,
                  run: bool = False, output: Path | None = None,
                  java: str = 'java', timeout: int = 180,
                  fixtures: tuple[Fixture,...]=FIXTURES) -> tuple[int,dict]:
    if not 20<=timeout<=600:
        raise ValueError('timeout must be 20..600 seconds')
    if len(fixtures)>20 or len({f.key for f in fixtures})!=len(fixtures):
        raise ValueError('fixture count/identifier invalid')
    preflight,jar_path=_jar_preflight(jar,expected_sha256)
    rows=[]
    # Local phase is always bounded and offline. Compiler phase starts only
    # after the entire fixed-fixture local Atlas baseline was evaluated.
    with tempfile.TemporaryDirectory(prefix='atlas-cljs-fixed-oracle-') as tmp:
        directory=Path(tmp)
        for index,fixture in enumerate(fixtures):
            fixture_root=directory/f'fixture{index:02d}'
            src=fixture_root/'src';src.mkdir(parents=True)
            checksums=_write_sources(src,fixture)
            local=_local_evidence(fixture,src)
            rows.append({'id':fixture.key,'compiler_expectation':fixture.compiler_expectation,
                         'reference_compiler_executed':False,'files_sha256':checksums,
                         'atlas':local})
        good_baseline=all(r['atlas']['atlas_candidate_confirmed'] for r in rows)
        report={'schema':SCHEMA,'documented_target_cljs_version':DOCUMENTED_CLJS_VERSION,
                'compiler_version_verified':False,'reference_binary_origin_attested':False,
                'user_project_code_executed':False,'automatic_network_downloads':False,
                'tested_fixtures_are_fixed_in_this_script':True,
                'compiler_conformance_scope':'fixed_explicit_cljs_macro_import_fixtures_only',
                'clojurescript_adapter_complete':False,
                'semantic_or_forensic_certified':False,
                'preflight':preflight,'run_opted_in':bool(run),
                'atlas_local_fixtures_passed':sum(r['atlas']['atlas_candidate_confirmed'] for r in rows),
                'reference_compiler_executed':False,'reference_passed_count':0,
                'reference_observational_count':0,'cases':rows}
        status=3
        if run and jar_path is not None and good_baseline:
            java_path=shutil.which(java)
            if java_path is None:
                report['not_executed_reason']='java_not_found'
            else:
                report['reference_compiler_executed']=True
                for i,(row,fixture) in enumerate(zip(rows,fixtures)):
                    result=_compiler_result(jar_path,java_path,fixture,
                                            directory/f'fixture{i:02d}',timeout)
                    row['reference_compiler_executed']=True
                    row['compiler']=result
                    row['positive_case_matches_compiler']=(
                        row['atlas']['atlas_candidate_confirmed'] and result['compiled']
                    ) if fixture.compiler_expectation=='compile_success' else None
                positive=[r for r in rows if r['compiler_expectation']=='compile_success']
                negative=[r for r in rows if r['compiler_expectation']=='compile_failure']
                controls_ok=bool(positive) and all(r['positive_case_matches_compiler'] for r in positive)
                report['negative_controls_eligible']=controls_ok
                report['reference_compiler_operational']=controls_ok
                report['reference_blocked_by_environment']=any(
                    r['compiler']['status']=='compiler_environment_error' for r in rows)
                for row in negative:
                    result=row['compiler']
                    row['negative_case_matches_compiler']=(
                        controls_ok and result.get('status')=='compiler_failed' and
                        result.get('compiler_exit_code',0)!=0)
                report['reference_negative_total']=len(negative)
                report['reference_negative_passed_count']=sum(
                    r['negative_case_matches_compiler'] for r in negative)
                observations=[r for r in rows if r['compiler_expectation']=='observe_only']
                report['reference_passed_count']=sum(r['positive_case_matches_compiler'] for r in positive)
                report['reference_positive_total']=len(positive)
                report['reference_observational_count']=len(observations)
                report['fixed_subset_compiler_concordance_passed']=(
                    good_baseline and len(positive)>0 and
                    report['reference_passed_count']==len(positive) and
                    report['reference_negative_passed_count']==len(negative))
                status=0 if report['fixed_subset_compiler_concordance_passed'] else 2
        if status==3:
            report['not_executed_reason']=report.get('not_executed_reason',
                'explicit_run_flag_missing' if not run else (
                    'atlas_local_baseline_mismatch' if not good_baseline else
                    preflight.get('reason','no_verified_local_jar')))
        if output is not None:
            if output.is_symlink():raise ValueError('refuse report symlink')
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return status,report


def main(argv: list[str] | None = None) -> int:
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',action='store_true',help='Explicitly enable local compiler execution on fixed fixtures only')
    cli.add_argument('--cljs-jar',type=Path,help='Trusted standalone cljs.jar, provided manually')
    cli.add_argument('--expected-sha256',help='SHA-256 of exactly the approved local standalone JAR')
    cli.add_argument('--java',default='java',help='Local Java executable, never a shell command')
    cli.add_argument('--timeout',type=int,default=180,help='Seconds per fixture, 20..600')
    cli.add_argument('--output',type=Path,default=Path('CLJS_COMPILER_REFERENCE_RESULT.json'))
    args=cli.parse_args(argv)
    try:
        status,report=run_reference(jar=args.cljs_jar,expected_sha256=args.expected_sha256,
                                    run=args.run,output=args.output,java=args.java,timeout=args.timeout)
    except (ValueError,PermissionError,OSError) as exc:
        print('REFERENCE NOT EXECUTED:',type(exc).__name__,str(exc)[:200],file=sys.stderr)
        return 4
    print('Atlas fixed local macro cases:',report['atlas_local_fixtures_passed'],'/',len(report['cases']))
    if report['reference_compiler_executed']:
        print('Actual CLJS compiler positive cases:',report['reference_passed_count'],
              '/',report['reference_positive_total'])
        print('Observational compiler cases:',report['reference_observational_count'])
    else:
        print('REFERENCE NOT EXECUTED:',report.get('not_executed_reason'))
    print('Report:',args.output)
    if status==0:
        print('ONLY the fixed compiler fixture subset passed. Whole adapter NOT certified.')
    elif status==2:
        print('COMPILER CONCORDANCE MISMATCH: inspect report, do not certify.')
    return status


if __name__=='__main__':
    sys.exit(main())
