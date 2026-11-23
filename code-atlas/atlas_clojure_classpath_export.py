#!/usr/bin/env python3
"""Convert an ALREADY RESOLVED classpath into a read-only Atlas snapshot.

Input is the literal output of a trusted `clojure -Spath` invocation saved by
its owner. Atlas never invokes that CLI or downloads any dependency. A snapshot
is evidence of supplied files, not independent verification of the originating
compiler command or guarantees about transitively selected versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def export(path_file, output, *, separator=None, base=None):
    infile=Path(path_file)
    if not infile.is_file() or infile.is_symlink() or infile.stat().st_size>150_000:
        raise ValueError('classpath_file_missing_unsafe_or_too_large')
    raw=infile.read_text(encoding='utf-8-sig').strip()
    if not raw or '\n' in raw or '\r' in raw:
        raise ValueError('expected_single_line_classpath')
    sep=separator or os.pathsep
    entries=[]
    root=Path(base or '.').resolve()
    for fragment in raw.split(sep):
        if not fragment:
            raise ValueError('empty_classpath_component')
        p=Path(fragment)
        if not p.is_absolute():p=root/p
        p=p.resolve()
        if p.is_symlink() or not p.exists():
            raise ValueError('missing_classpath_component:'+fragment[:80])
        kind='directory' if p.is_dir() else 'jar' if p.is_file() and p.suffix.lower()=='.jar' else None
        if kind is None:
            raise ValueError('classpath_entry_not_jar_or_directory')
        record={'kind':kind,'path':str(p)}
        if kind=='jar':
            h=hashlib.sha256()
            with p.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):
                    h.update(chunk)
            record['sha256']=h.hexdigest()
        entries.append(record)
    if len(entries)>256:
        raise ValueError('too_many_classpath_entries')
    output=Path(output)
    if output.resolve()==infile.resolve():
        raise ValueError('output_must_not_overwrite_classpath_source')
    data={'schema':'atlas-clojure-classpath-v1',
          'source':'user_supplied_pre_resolved_classpath_no_compiler_invocation',
          'entries':entries}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--classpath-file',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--separator',choices=(':',';'),default=None,
                    help='Windows export použij ;. Předvolený oddělovač je lokální OS.')
    ap.add_argument('--base',type=Path,default=None,
                    help='Pracovní složka, vůči které byl vytvořen classpath soubor')
    args=ap.parse_args()
    data=export(args.classpath_file,args.output,separator=args.separator,base=args.base)
    print(f'Zapsaný lokální snapshot: {len(data["entries"])} položek. Žádná síť ani build.')

if __name__=='__main__':main()
