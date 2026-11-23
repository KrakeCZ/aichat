"""Read-only, explicitly selected, local ClojureScript compiler core index.

Analyzes only trusted cljs/core.cljs from the pinned local cljs.jar in an
isolated JVM. Source code from an analyzed project never reaches the compiler.
This is a real cljs.analyzer.api namespace index, not type inference for user
code or validation of a project's classpath and macro environment.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from functools import lru_cache

from .clojure_core_oracle import NAME_RE, MAX_REQUESTS
from .clojure_official_reader import check_local_jars


@lru_cache(maxsize=3)
def _full_core_index(signature, paths, harness_path, timeout):
    """Compile the pinned cljs/core.cljs once per immutable JAR identity."""
    java=shutil.which('java')
    if not java:raise ValueError('java_missing_for_cljs_core_oracle')
    cmd=[java,'-Xms32m','-Xmx768m','-XX:ActiveProcessorCount=2',
         '-Dclojure.main.report=stderr','-cp',os.pathsep.join(paths),
         'clojure.main',harness_path]
    env={k:v for k,v in os.environ.items() if k in
         ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','TMPDIR')}
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-safe-cljs-core-') as tmp:
            result=subprocess.run(cmd,cwd=tmp,env=env,input='\n',
                                  capture_output=True,text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        raise ValueError('cljs_core_oracle_timeout') from None
    if result.returncode:raise ValueError('cljs_core_oracle_subprocess_failed')
    observed={};version=None;expected_count=None
    for line in result.stdout.splitlines():
        fields=line.split('\t')
        if len(fields)==3 and fields[0]=='ATLAS_CLJS_CORE_ORACLE':
            version=fields[1]
            try:expected_count=int(fields[2])
            except ValueError:raise ValueError('cljs_core_oracle_bad_count') from None
            continue
        if not fields or fields[0]!='ATLAS_CLJS_CORE_INDEX':continue
        if len(fields)!=6:raise ValueError('cljs_core_oracle_bad_index_line')
        name,macro,fn_var,tag,arity=fields[1:]
        if name in observed or len(name)>200 or any(ord(ch)<32 for ch in name) or macro not in ('true','false') or fn_var not in ('true','false'):
            raise ValueError('cljs_core_oracle_bad_index_member')
        try:
            hint=base64.b64decode(tag,validate=True).decode('utf8')
            shape=base64.b64decode(arity,validate=True).decode('ascii')
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
        except (ValueError,UnicodeError):raise ValueError('cljs_core_oracle_bad_metadata') from None
        observed[name]={'status':'found','qualified_name':'cljs.core/'+name,
                         'macro':macro=='true','fn_var':fn_var=='true',
                         'compiler_tag':hint or None,
                         'declared_fixed_arities':fixed,
                         'declared_variadic_min_arity':variadic,
                         'arity_metadata_available':bool(shape)}
    if version is None or expected_count is None or len(observed)!=expected_count:
        raise ValueError('cljs_core_oracle_incomplete_index')
    return version,expected_count,observed


def cljs_core_oracle(names, jar_dir:Path, *, timeout=75):
    names=sorted(set(names))
    if len(names)>MAX_REQUESTS or any(not isinstance(n,str) or not NAME_RE.fullmatch(n) for n in names):
        raise ValueError('cljs_core_oracle_names_invalid_or_too_many')
    jars=check_local_jars(Path(jar_dir))
    harness=Path(__file__).resolve().parents[2]/'reference'/'atlas_clojurescript_safe_core_oracle.clj'
    signature=tuple(j['sha256'] for j in jars)+(hashlib.sha256(harness.read_bytes()).hexdigest(),)
    version,count,index=_full_core_index(signature,tuple(j['path'] for j in jars),str(harness),timeout)
    missing={'status':'missing','qualified_name':None,'macro':False,
             'fn_var':False,'compiler_tag':None,
             'declared_fixed_arities':[],'declared_variadic_min_arity':None,
             'arity_metadata_available':False}
    return {'status':'trusted_cljs_core_compiler_index','provenance':{
                'backend':'cljs.analyzer.api/ns-publics on trusted cljs/core.cljs',
                'clojure_host_version':version,'core_definition_count':count,
                'jar_sha256':{Path(j['path']).name:j['sha256'] for j in jars},
                'harness_sha256':signature[-1],
                'project_namespace_loaded':False,'project_source_analyzed':False},
            'vars':{name:index.get(name,missing) for name in names}}
