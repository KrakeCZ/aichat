#!/usr/bin/env python3
"""Opt-in pinned local JVM oracle for embedded namespace/privacy fixtures only.

No user source/path, downloader, REPL or build configuration is accepted.
The JVM loads only the two embedded synthetic lib files and case strings.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from analyzer.semantic_adapters.clojure_forms import read_forms, _namespace, _explicit_refers, _unresolved_libspecs, _definition, _callable_visibility
from atlas_clojure_reference import resolve_jar_dir, preflight_jars

CASES=(
 ('prefix_list','(:require (lib [a :as a] [b :refer [g]]))','',True),
 ('prefix_vector','(:require [lib [a :as a] [b :refer [g]]])','',True),
 ('prefix_rename','(:require (lib [b :refer [g] :rename {g renamed}]))','',True),
 ('prefix_bare','(:require (lib a b))','',True),
 ('dotted_child','(:require (lib [a.part :as a]))','',False),
 ('boolean_alias','(:require [lib.a :as true])','',False),
 ('private_leading','','(defn hidden {:private true} [] 1)',True),
 ('private_trailing','','(defn hidden ([x] x) {:private true})',True),
 ('private_override','','(defn ^:private hidden {:private false} [] 1)',True),
 ('private_macro','','(defmacro hidden {:private true} [] 1)',True),
)
PROVIDERS={'lib/a.clj':'(ns lib.a) (defn f [] 1)', 'lib/b.clj':'(ns lib.b) (defn g [] 2)'}
HARNESS=r'''
(require '[clojure.string :as str])
(println (str "ATLAS_NS_VERSION\t" (clojure-version)))
(doseq [line (line-seq (java.io.BufferedReader. *in*))]
 (let [[id encoded] (str/split line #"\t")
       source (String. (.decode (java.util.Base64/getDecoder) encoded) "UTF-8")]
  (try
   (binding [*ns* *ns* *read-eval* false] (load-string source)
    (let [target-ns (the-ns (symbol (str "atlas.fixture." id)))
          aliases (str/join "," (sort (map (fn [[a n]] (str a "=" (ns-name n))) (ns-aliases target-ns))))
          refs (str/join "," (sort (for [[n v] (ns-refers target-ns) :let [target (str (ns-name (:ns (meta v))))] :when (.startsWith target "lib.")]
                               (str n "=" target "/" (:name (meta v))))))
          v (ns-resolve target-ns 'hidden)
          privacy (if v (if (:private (meta v)) "private" "public") "none")]
     (println (str "ATLAS_NS\t" id "\taccepted\t" aliases "\t" refs "\t" privacy))))
   (catch Throwable e (println (str "ATLAS_NS\t" id "\trejected\t\t\tnone"))))))
'''

def source_for(case):
    key,clause,definition,_=case
    return f'(ns atlas.fixture.{key} {clause})\n{definition}'

def local_result(case):
    forms=read_forms(source_for(case));ns,aliases,_,issue=_namespace(forms)
    invalid=_unresolved_libspecs(forms)
    refers,_=_explicit_refers(forms)
    private='none'
    for form in forms:
        if _definition(form):private=_callable_visibility(form)[0]
    return ['accepted' if ns and not issue and not invalid else 'rejected',
        ','.join(sorted(f'{a}={n}' for a,n in aliases.items())),
        ','.join(sorted(f'{name}={v[0]}/{v[1]}' for name,v in refers.items() if v)),private]

def run(jar_dir=None):
    local={case[0]:local_result(case) for case in CASES}
    report={'schema':'atlas-fixed-namespace-privacy-reference/1','user_project_code_executed':False,
        'automatic_downloads':False,'reference_runtime_executed':False,'adapter_complete':False,
        'fixtures_sha256':hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
        'harness_sha256':hashlib.sha256(HARNESS.encode()).hexdigest(),
        'providers_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in PROVIDERS.items()},
        'source_sha256':{c[0]:hashlib.sha256(source_for(c).encode()).hexdigest() for c in CASES},
        'cases':[{'id':c[0],'atlas':local[c[0]],'expected_accepted':c[3]} for c in CASES]}
    if jar_dir is None:return 3,report
    jars=resolve_jar_dir(jar_dir);report['jars']=preflight_jars(jars)
    java=shutil.which('java')
    if not java:raise ValueError('java not found')
    with tempfile.TemporaryDirectory(prefix='atlas-fixed-namespaces-') as temp:
        root=Path(temp)
        for name,source in PROVIDERS.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source,encoding='utf8')
        harness=root/'oracle.clj';harness.write_text(HARNESS,encoding='utf8')
        payload=''.join(c[0]+'\t'+base64.b64encode(source_for(c).encode()).decode()+'\n' for c in CASES)
        env={k:v for k,v in os.environ.items() if k in ('PATH','SystemRoot','SYSTEMROOT','WINDIR','TEMP','TMP','HOME','USERPROFILE')}
        cp=os.pathsep.join([*(str(p.resolve()) for p in jars),str(root)])
        result=subprocess.run([java,'-Xmx256m','-Duser.home='+temp,'-cp',cp,'clojure.main',str(harness)],
            input=payload,text=True,capture_output=True,timeout=45,cwd=root,env=env,check=False)
        if result.returncode:raise RuntimeError(result.stderr[-1200:])
    rows={};version=None
    for line in result.stdout.splitlines():
        fields=line.split('\t')
        if fields[0]=='ATLAS_NS_VERSION':version=fields[1]
        if fields[0]=='ATLAS_NS' and len(fields)==6:rows[fields[1]]=fields[2:]
    report.update(reference_runtime_executed=True,reference_version=version)
    for row in report['cases']:
        actual=rows.get(row['id']);row['jvm']=actual
        row['matches']=bool(actual and actual[0]==row['atlas'][0] and
            (actual[0]=='rejected' or actual==row['atlas']) and
            (actual[0]=='accepted')==row['expected_accepted'])
    report['passed']=sum(r['matches'] for r in report['cases']);report['total']=len(CASES)
    return (0 if report['passed']==len(CASES) and version=='1.12.6' else 2),report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jar-dir',type=Path)
    p.add_argument('--output',type=Path,default=Path('STAGE48_NAMESPACE_REFERENCE.json'));args=p.parse_args()
    code,report=run(args.jar_dir)
    if args.output.is_symlink():raise ValueError('report symlink refused')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('Namespace reference:',report.get('passed',0),'/',len(CASES),'exit',code)
    return code
if __name__=='__main__':raise SystemExit(main())
