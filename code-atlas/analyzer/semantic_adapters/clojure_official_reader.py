"""Explicit offline Clojure 1.12.6 core-reader / CLJS tools.reader validation.

This is a reader validation tier, not a type checker or macro expander. Project
source NEVER appears in the JVM classpath; the JVM has a fixed classpath and a
limited subprocess lifetime. No runtime dependency resolution is attempted.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

READER_JARS = ('clojure-1.12.6.jar', 'spec.alpha-0.5.238.jar',
               'core.specs.alpha-0.4.74.jar', 'cljs.jar')
MAX_READER_BATCH = 300
MAX_READER_BYTES = 750_000


def check_local_jars(directory: Path):
    """Return pinned file metadata, without running any Java process."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('clojure_reader_jar_dir_missing_or_symlink')
    jars = []
    for name in READER_JARS:
        jar = directory / name
        if jar.is_symlink() or not jar.is_file() or not jar.resolve().is_relative_to(directory.resolve()):
            raise ValueError('clojure_reader_jar_missing_or_unsafe:'+name)
        if not (0 < jar.stat().st_size <= 80_000_000):
            raise ValueError('clojure_reader_jar_size_invalid:'+name)
        try:
            with zipfile.ZipFile(jar) as z:
                if len(z.namelist()) > 100_000 or z.testzip() is not None:
                    raise ValueError('clojure_reader_jar_corrupt:'+name)
        except zipfile.BadZipFile:
            raise ValueError('clojure_reader_jar_invalid_zip:'+name) from None
        jars.append({'path':str(jar.resolve()),'sha256':hashlib.sha256(jar.read_bytes()).hexdigest()})
    return jars


def validate_sources(sources: list[tuple[str,str,str]], jar_dir: Path, *, timeout=25):
    """Return record keyed by index. Sources are (path, platform, text) tuples.

    `cljc-cljs` is an oracle-only dialect selector for .cljc :cljs features;
    `cljc` selects JVM :clj. `cljs-compiler` and `cljc-compiler-cljs`
    enable ONLY the four pinned, trusted built-in CLJS tagged-literal readers
    on inert QA fixtures. Never pass an analyzed project to this oracle.

    Only an explicit user opt-in enables this backend. Each batch uses a new
    JVM with fixed args, memory ceiling, no external classpath or network
    client. The caller must keep batch size and time bounded.
    """
    if len(sources)>MAX_READER_BATCH:
        raise ValueError('clojure_reader_batch_limit')
    jars=check_local_jars(jar_dir)
    java=shutil.which('java')
    if not java:
        raise ValueError('java_not_locally_installed')
    harness=Path(__file__).resolve().parents[2]/'reference'/'atlas_clojure_safe_source_syntax.clj'
    payload=[]
    for index, (_,platform,text) in enumerate(sources):
        if platform not in ('clj','cljs','cljc','cljc-cljs','cljs-compiler','cljc-compiler-cljs') or len(text.encode('utf-8'))>MAX_READER_BYTES:
            raise ValueError('source_platform_or_size_not_supported')
        payload.append(f'{index}\t{platform}\t{base64.b64encode(text.encode("utf-8")).decode("ascii")}')
    env={k:v for k,v in os.environ.items() if k in
         ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR')}
    # This limits Java heap, but is not a hardened OS-level sandbox.
    cmd=[java,'-Xms16m','-Xmx256m','-XX:ActiveProcessorCount=2',
         '-Dclojure.main.report=stderr','-cp',os.pathsep.join(j['path'] for j in jars),
         'clojure.main',str(harness)]
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-offline-clj-reader-') as scratch:
            result=subprocess.run(cmd,input='\n'.join(payload)+'\n',text=True,
                                  capture_output=True,cwd=scratch,env=env,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        raise ValueError('official_reader_oracle_timeout') from None
    if result.returncode:
        raise ValueError('official_reader_oracle_unavailable')
    lines=result.stdout.splitlines()
    version=next((x.split('\t',1)[1] for x in lines if x.startswith('ATLAS_SOURCE_READER\t')),None)
    observations={}
    for line in lines:
        parts=line.split('\t')
        if len(parts)!=5 or parts[0]!='ATLAS_SOURCE_RESULT':
            continue
        try:index=int(parts[1]);n=int(parts[3])
        except ValueError:continue
        if (index in observations or index>=len(sources) or index<0 or
                parts[2] not in ('accepted','rejected') or
                parts[4] not in ('clojure.core/read','cljs.vendor.clojure.tools.reader',
                                  'cljs.tagged-literals/builtins+vendor.tools.reader')):
            raise ValueError('invalid_official_reader_response')
        observations[index]={'status':parts[2], 'forms_read':n,
                             'reader':parts[4], 'clojure_version':version,
                             'source_sha256':hashlib.sha256(sources[index][2].encode('utf-8')).hexdigest(),
                             'jar_sha256':{Path(j['path']).name:j['sha256'] for j in jars},
                             'note':'safe_reader_validation_only_no_macro_or_type_resolution'}
    if len(observations)!=len(sources) or not version:
        raise ValueError('incomplete_official_reader_response')
    return {sources[i][0]:observations[i] for i in range(len(sources))}
