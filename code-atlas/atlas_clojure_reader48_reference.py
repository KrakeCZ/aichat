#!/usr/bin/env python3
"""Opt-in exact reader acceptance/count comparison for Stage48 embedded data."""
import argparse
import hashlib
import json
from pathlib import Path
from analyzer.semantic_adapters.clojure_forms import read_forms, ReaderError
from atlas_clojure_reference import run_reference, resolve_jar_dir, preflight_jars

FIXTURES=(
 ('dispatch_comment','clj','(ns app)\n#! comment (defn fake [] 1)\n(defn real [] 1)',True,2),
 ('nested_discard','clj','#_ #_ :a :b :kept',True,1),
 ('vector_nested_discard','clj','[#_ #_ :a :b :kept]',True,1),
 ('quote_discard','clj',"' #_ :a :kept",True,1),
 ('metadata_discard','clj','^{:doc "x"} #_ :a kept',True,1),
 ('incomplete_nested_discard','clj','#_ #_ :a',False,0),
 ('incomplete_vector_discard','clj','[#_ #_ :a]',False,0),
 ('incomplete_quote_discard','clj',"' #_ :a",False,0),
 ('equivalent_sequence_map','clj','{[1] :a (1) :b}',False,0),
 ('equivalent_sequence_set','clj','#{[1] (1)}',False,0),
 ('nested_sequence_equality','clj','#{[[1]] ((1))}',False,0),
 ('distinct_sequences','clj','#{[1] (2)}',True,1),
 ('distinct_types','clj','#{[true] [1]}',True,1),
)

def run(folder=None):
    rows=[]
    for key,_,source,accepted,count in FIXTURES:
        try:local={'accepted':True,'forms':len(read_forms(source))}
        except ReaderError:local={'accepted':False,'forms':0}
        rows.append({'id':key,'expected':{'accepted':accepted,'forms':count},'atlas':local})
    report={'schema':'atlas-reader48-reference/1','project_code_executed':False,'network_requested':False,
        'reference_runtime_executed':False,'complete_language_conformance':False,
        'fixture_sha256':hashlib.sha256(json.dumps(FIXTURES).encode()).hexdigest(),'cases':rows}
    if folder is None:return 3,report
    jars=resolve_jar_dir(folder);report['jars']=preflight_jars(jars)
    version,results,_=run_reference(jars,'java',45,fixtures=FIXTURES)
    report.update(reference_runtime_executed=True,reference_version=version)
    for row in rows:
        row['jvm']=results[row['id']]
        row['matches']=row['jvm']==row['atlas']==row['expected']
    report['passed']=sum(r['matches'] for r in rows);report['total']=len(rows)
    return (0 if report['passed']==len(rows) else 2),report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jar-dir',type=Path)
    p.add_argument('--output',type=Path,default=Path('STAGE48_READER_REFERENCE.json'));a=p.parse_args()
    code,report=run(a.jar_dir)
    if a.output.is_symlink():raise ValueError('report symlink refused')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('Reader48:',report.get('passed',0),'/',len(FIXTURES),'exit',code)
    return code
if __name__=='__main__':raise SystemExit(main())
