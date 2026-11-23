"""Compiler-runtime backed Clojure core Var metadata and audited core macro expansion.

Only Clojure's pinned standard-library JARs enter the classpath. Project code
is passed as UTF-8 reader data, never compiled, evaluated or required. This
provides genuine clojure.core Var identity and core macro transformations;
it does NOT resolve arbitrary project Vars, cljs.core, types or dynamic calls.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .clojure_official_reader import check_local_jars

SAFE_MACROS = frozenset({'when','when-not','if-not','and','or','->','->>','cond'})
MAX_REQUESTS = 2000
MAX_BYTES = 32_000
NAME_RE = re.compile(r'^[A-Za-z0-9*+!_?=<>./:-]{1,109}$')


def core_oracle(names, expansions, jar_dir: Path, *, timeout=35):
    """Query compiler-runtime core vars and expand selected built-in macros.

    `names`: iterable of unqualified Var names. `expansions`: iterable of
    (name, source) pairs. Non-whitelisted macros or arbitrary project code
    are never executed. The result contains authenticated provenance hashes.
    """
    names=sorted(set(names))
    expansions=list(expansions)
    if len(names)+len(expansions)>MAX_REQUESTS:
        raise ValueError('core_oracle_request_limit')
    if any(not isinstance(x,str) or not NAME_RE.fullmatch(x) for x in names):
        raise ValueError('core_oracle_invalid_var_name')
    if any((not isinstance(n,str) or n not in SAFE_MACROS or
            not isinstance(s,str) or len(s.encode('utf-8'))>MAX_BYTES)
           for n,s in expansions):
        raise ValueError('core_oracle_invalid_macro_request')
    jars=check_local_jars(Path(jar_dir))
    java=shutil.which('java')
    if not java:raise ValueError('java_not_installed_for_core_oracle')
    harness=(Path(__file__).resolve().parents[2]/'reference'/'atlas_clojure_safe_core_oracle.clj')
    def encoded(s):return base64.b64encode(s.encode('utf-8')).decode('ascii')
    messages=[f'S\t{i}\t{encoded(name)}' for i,name in enumerate(names)]
    messages += [f'M\t{i+len(names)}\t{encoded(n)}\t{encoded(s)}'
                 for i,(n,s) in enumerate(expansions)]
    env={k:v for k,v in os.environ.items() if k in
         ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR')}
    command=[java,'-Xms16m','-Xmx256m','-XX:ActiveProcessorCount=2',
             '-Dclojure.main.report=stderr', '-cp',os.pathsep.join(j['path'] for j in jars),
             'clojure.main',str(harness)]
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-safe-core-oracle-') as tmp:
            result=subprocess.run(command,input='\n'.join(messages)+'\n',text=True,
                                  capture_output=True,cwd=tmp,env=env,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        raise ValueError('core_oracle_timeout') from None
    if result.returncode:
        raise ValueError('core_oracle_subprocess_failed')
    output={};clojure_version=None
    for line in result.stdout.splitlines():
        if line.startswith('ATLAS_CORE_ORACLE\t'):
            clojure_version=line.split('\t',1)[1];continue
        fields=line.split('\t')
        if len(fields)<5 or fields[0]!='ATLAS_CORE_RESULT':continue
        try:index=int(fields[1])
        except ValueError:continue
        if not 0<=index<len(messages) or index in output:raise ValueError('core_oracle_invalid_result_index')
        kind=fields[2]
        if index<len(names):
            if kind!='S' or len(fields)!=8:raise ValueError('core_oracle_bad_symbol_record')
            status,qual,macro,tag,arity=fields[3:]
            if status not in ('found','missing') or macro not in ('true','false'):
                raise ValueError('core_oracle_bad_symbol_status')
            try:
                decoded=base64.b64decode(tag,validate=True).decode('utf8') if status=='found' else ''
                shape=base64.b64decode(arity,validate=True).decode('ascii') if status=='found' else ''
                if shape:
                    fixed_text,variadic_text=shape.split(';')
                    fixed=[int(x) for x in fixed_text.split(',') if x]
                    variadic=int(variadic_text) if variadic_text else None
                    if (len(fixed)>40 or sorted(set(fixed))!=fixed or
                            any(not 0<=x<=255 for x in fixed) or
                            (variadic is not None and not 0<=variadic<=255)):
                        raise ValueError('invalid arity')
                else:
                    fixed=[];variadic=None
            except (ValueError,UnicodeError):raise ValueError('core_oracle_bad_metadata') from None
            output[index]={'status':status,'qualified_name':qual if status=='found' else None,
                           'macro':macro=='true','compiler_var_tag':decoded or None,
                           'declared_fixed_arities':fixed,
                           'declared_variadic_min_arity':variadic,
                           'arity_metadata_available':bool(shape)}
        else:
            if kind!='M' or len(fields)!=5:raise ValueError('core_oracle_bad_macro_record')
            status=fields[3]
            if status not in ('expanded','unsupported','too_large','rejected'):
                raise ValueError('core_oracle_bad_macro_status')
            expanded=None
            if status=='expanded':
                try:expanded=base64.b64decode(fields[4],validate=True).decode('utf8')
                except (ValueError,UnicodeError):raise ValueError('core_oracle_bad_expansion') from None
            output[index]={'status':status,'expanded_syntax':expanded}
    if not clojure_version or len(output)!=len(messages):
        raise ValueError('core_oracle_incomplete_response')
    provenance={'backend':'clojure.core runtime Var registry',
                'clojure_version':clojure_version,
                'jar_sha256':{Path(j['path']).name:j['sha256'] for j in jars},
                'harness_sha256':hashlib.sha256(harness.read_bytes()).hexdigest(),
                'project_namespace_loaded':False,'project_macro_invoked':False}
    return {'status':'trusted_pinned_clojure_core_only', 'provenance':provenance,
            'vars':{name:output[i] for i,name in enumerate(names)},
            'expansions':[{'macro':n,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),
                           **output[len(names)+i]}
                          for i,(n,s) in enumerate(expansions)]}
