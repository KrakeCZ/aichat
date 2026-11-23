"""Commit-by-commit architecture evidence for Universal Code Atlas.

Never executes the analyzed Git project's code and never checks out commits.
Each selected revision is extracted into a temporary tree, analyzed with the
same project-neutral adapters, and compared by typed file-level relationships.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from typing import Any, Callable


MAX_COMMITS = 60
MAX_FILES = 7000
MAX_TOTAL_SOURCE_BYTES = 80 * 1024 * 1024
MAX_FILE_BYTES = 3 * 1024 * 1024


def _git(repo: Path, *args: str, timeout: int = 40) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise ValueError("Git operace selhala: " + proc.stderr.decode('utf-8', 'replace')[:350])
    return proc.stdout


def repository_root(project: Path) -> Path:
    root = _git(project, 'rev-parse', '--show-toplevel').decode('utf-8', 'replace').strip()
    result = Path(root).resolve()
    if not result.is_dir():
        raise ValueError("Git root neexistuje")
    return result


def selected_commits(repo: Path, ref: str = 'HEAD', limit: int = 8, stride: int = 1) -> list[dict]:
    if not 1 <= limit <= MAX_COMMITS or not 1 <= stride <= 1000:
        raise ValueError(f"Limit musí být 1–{MAX_COMMITS}; stride 1–1000")
    if not re.fullmatch(r'[A-Za-z0-9_./@{}^~+\-]+', ref) or ref.startswith('-'):
        raise ValueError("Nepovolený Git ref")
    commit = _git(repo, 'rev-parse', '--verify', '--end-of-options', ref+'^{commit}').decode().strip()
    raw = _git(repo, 'rev-list', '--first-parent', '--max-count='+str(limit*stride), commit)
    candidates = [x.strip() for x in raw.decode().splitlines() if x.strip()]
    shas = list(reversed(candidates[::stride][:limit]))
    output = []
    for sha in shas:
        meta = _git(repo, 'show', '-s', '--format=%H%x00%ct%x00%s%x00%an', sha)
        full, epoch, title, author = (meta.decode('utf-8', 'replace').strip().split('\0') + ['']*4)[:4]
        output.append({"sha":full, "time_utc":datetime.fromtimestamp(int(epoch),timezone.utc).isoformat(),
                       "title":title[:200], "author":author[:100]})
    return output


def _safe_relative(name: str) -> Path | None:
    if not name or '\\' in name or '\x00' in name:
        return None
    pp = PurePosixPath(name)
    if pp.is_absolute() or any(part in ('..', '.git') for part in pp.parts):
        return None
    return Path(*pp.parts)


def extract_revision(repo: Path, sha: str, target: Path) -> dict:
    """Read-only git archive; never extract symlinks or archives' own paths blindly."""
    proc = subprocess.Popen(['git','-C',str(repo),'archive','--format=tar',sha],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    extracted = 0
    total = 0
    skipped_large = []
    skipped_links = []
    lfs_pointers = []
    gitlinks = []
    for entry in _git(repo,'ls-tree','-r','-z',sha,timeout=45).split(b'\0'):
        if entry.startswith(b'160000 '):
            gitlinks.append(entry.split(b'\t',1)[-1].decode('utf-8','replace'))
    try:
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode='r|') as stream:
            for member in stream:
                if not member.isfile():
                    if member.issym() or member.islnk():
                        skipped_links.append(member.name)
                    continue  # no symlinks, hard links, submodules or special files
                rel = _safe_relative(member.name)
                if rel is None:
                    raise ValueError('Nebezpečná cesta v Git stromu')
                if member.size > MAX_FILE_BYTES:
                    skipped_large.append(str(rel))
                    continue
                extracted += 1
                total += member.size
                if extracted > MAX_FILES or total > MAX_TOTAL_SOURCE_BYTES:
                    raise ValueError('Git snapshot překročil bezpečný limit velikosti / souborů')
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                reader = stream.extractfile(member)
                if reader is None:
                    continue
                with dest.open('wb') as writer:
                    remaining = member.size
                    while remaining:
                        data = reader.read(min(65536, remaining))
                        if not data:
                            raise ValueError('Neočekávaný konec Git archivu')
                        if remaining == member.size and data.startswith(b'version https://git-lfs.github.com/spec/v1'):
                            lfs_pointers.append(str(rel))
                        writer.write(data)
                        remaining -= len(data)
        code = proc.wait(timeout=15)
        if code:
            err = proc.stderr.read(450).decode('utf-8','replace') if proc.stderr else ''
            raise ValueError(f'Git archive selhal: {err}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
    return {'files_extracted': extracted, 'bytes_extracted': total,
            'skipped_large': skipped_large, 'skipped_links':skipped_links,
            'submodule_gitlinks':gitlinks, 'lfs_pointers':lfs_pointers,
            'complete': not (skipped_large or skipped_links or gitlinks or lfs_pointers)}


def edge_id(edge: dict) -> tuple[str,str,str,str]:
    return (str(edge.get('source','')), str(edge.get('target','')),
            str(edge.get('type','')), str(edge.get('confidence_v3') or edge.get('confidence') or 'unknown'))


def compact_edge(edge: dict) -> dict:
    ev=[]
    for item in edge.get('evidence',[]):
        ev.append({'path':item.get('path',''), 'line':item.get('line',0), 'column':item.get('column',0),
                   'snippet':str(item.get('snippet') or item.get('text') or '')[:240]})
    return {'source':edge['source'], 'target':edge['target'], 'type':edge.get('type',''),
            'confidence':edge.get('confidence_v3') or edge.get('confidence') or 'unknown',
            'analyzer':edge.get('analyzer',''), 'evidence':ev}


def snapshot_graph(graph: dict, symbol_index: dict, sha: str) -> dict:
    nodes = sorted({n.get('id') for n in graph.get('nodes',[]) if n.get('type') == 'file' and n.get('id')})
    edges = {edge_id(e):compact_edge(e) for e in graph.get('edges',[]) if e.get('source') and e.get('target')}
    parsers=Counter((v.get('parser') or 'unknown') for v in symbol_index.get('files',{}).values())
    return {'sha':sha, 'files':nodes, 'edges':sorted(edges.values(),key=lambda e:edge_id(e)),
            'symbol_count':sum(int(v.get('symbol_count') or 0) for v in symbol_index.get('files',{}).values()),
            'parser_counts':dict(sorted(parsers.items())), 'analysis_errors':symbol_index.get('errors', [])}


def diff_snapshots(before: dict | None, after: dict) -> dict:
    left=set(before['files']) if before else set()
    right=set(after['files'])
    a={edge_id(e):e for e in before['edges']} if before else {}
    b={edge_id(e):e for e in after['edges']}
    added=[b[key] for key in sorted(b.keys()-a.keys())]
    removed=[a[key] for key in sorted(a.keys()-b.keys())]
    changed_evidence=[{'before':a[k],'after':b[k]} for k in sorted(a.keys() & b.keys())
                      if a[k].get('evidence')!=b[k].get('evidence')]
    return {'changed_evidence':changed_evidence, 'added_files': sorted(right-left), 'removed_files':sorted(left-right),
            'added_edges':added, 'removed_edges':removed,
            'files':len(after['files']), 'edges':len(after['edges']),
            'symbols':after['symbol_count']}


def git_name_changes(repo: Path, previous_sha: str | None, sha: str) -> list[dict]:
    if not previous_sha:
        return []
    raw=_git(repo,'diff','--name-status','-z','-M',previous_sha,sha,timeout=60)
    bits=raw.decode('utf-8','replace').split('\0')
    changes=[]
    i=0
    while i<len(bits) and bits[i]:
        status=bits[i]; i+=1
        if status.startswith(('R','C')):
            if i+1>=len(bits): break
            old,new=bits[i:i+2];i+=2
            changes.append({'status':status[0],'old':old,'path':new})
        else:
            if i>=len(bits): break
            path=bits[i]; i+=1
            changes.append({'status':status[0],'path':path})
    return changes


def make_timeline(project: Path, output: Path, *, ref: str, limit: int, stride: int,
                  repository: str, prefix: str,
                  analyze_fn: Callable[[Path,Path,str,str,bool],Any], verbose: bool = True) -> dict:
    root=repository_root(project.resolve())
    output=output.expanduser().resolve()
    if output==root or output==root.parent or output==Path(__file__).resolve().parent.parent:
        raise ValueError('Výstup nesmí přepsat zdrojový strom ani instalaci Atlasu')
    commits=selected_commits(root,ref,limit,stride)
    if not commits:
        raise ValueError('Git neobsahuje žádné commity')
    result={'schema':'atlas-git-timeline/1', 'project':root.name,'ref':ref,
            'sample_stride':stride, 'repo':repository, 'note':
            'Každý bod je samostatná statická analýza Git snapshotu; žádný commit ani analyzovaný kód se nespouští. '
            'Časové změny nejsou důkazem runtime příčinnosti.', 'commits':[]}
    prev_snapshot=None
    prev_sha=None
    with tempfile.TemporaryDirectory(prefix='uca-timeline-') as tmp:
        tmpdir=Path(tmp)
        for i,info in enumerate(commits):
            snapshot_dir=tmpdir/root.name
            snapshot_dir.mkdir()
            snapshot_info=extract_revision(root,info['sha'],snapshot_dir)
            site=output if i==len(commits)-1 else tmpdir/'site'
            if verbose:
                print(f'[history {i+1}/{len(commits)}] {info["sha"][:10]} {info["title"]}',flush=True)
            analyze_fn(snapshot_dir,site,repository,prefix,False)
            graph=json.loads((site/'graph-index.json').read_text(encoding='utf-8'))
            sym=json.loads((site/'symbol-index.json').read_text(encoding='utf-8'))
            current=snapshot_graph(graph,sym,info['sha'])
            change=diff_snapshots(prev_snapshot,current)
            entry={**info, 'snapshot':{'files':len(current['files']), 'edges':len(current['edges']),
                                        'symbols':current['symbol_count'],
                                        'parser_counts':current['parser_counts'],
                                        'analysis_errors':current['analysis_errors'],
                                        'extraction':snapshot_info},
                   'delta':change, 'git_changes':git_name_changes(root,prev_sha,info['sha']),
                   'edges':current['edges'], 'files':current['files']}
            result['commits'].append(entry)
            # Main site corresponds exactly to the newest commit; source files are NOT published.
            if i==len(commits)-1:
                for indexname in ('graph-index.json','source-index.json','symbol-index.json'):
                    path=site/indexname
                    doc=json.loads(path.read_text(encoding='utf-8'))
                    doc['source_commit']=info['sha']
                    doc['source_mode']='git_snapshot'
                    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            prev_snapshot=current
            prev_sha=info['sha']
            # After analysis, remove old intermediate site, never touch the user's repository.
            import shutil
            if site!=output:
                shutil.rmtree(site)
            shutil.rmtree(snapshot_dir)
    (output/'git-timeline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
