#!/usr/bin/env python3
"""Refresh offline toolchain version evidence in an existing Atlas encyclopedia.

No compiler, npm, Cargo, Gradle, Maven, dotnet or project commands are run.
Source/symbol/graph indexes remain untouched; only static panel/JS/report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyzer.toolchain_versions import inspect_toolchain_versions

BASE = Path(__file__).resolve().parent


def refresh(project: Path, output: Path) -> dict:
    root = project.expanduser().resolve()
    site = output.expanduser().resolve()
    if not root.is_dir() or not site.is_dir() or root == site or site == BASE:
        raise ValueError('Vyžaduje existující odlišný repozitář a existující výstup Atlasu.')
    index, page, portal = (site / name for name in ('source-index.json','index.html','portal.js'))
    if not all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(site)
               for p in (index,page,portal)):
        raise ValueError('Poškozený Atlas nebo nepovolený symbolický odkaz.')
    data = json.loads(index.read_text(encoding='utf-8'))
    paths = {x['path'] for x in data.get('files',[]) if isinstance(x,dict) and isinstance(x.get('path'),str)}
    report = inspect_toolchain_versions(root,paths)
    report_path=site/'toolchain-version-report.json'
    if report_path.exists() and (report_path.is_symlink() or not report_path.resolve().is_relative_to(site)):
        raise ValueError('Nepovolený výstupní cíl reportu.')
    source_page=(BASE/'views/index.html').read_text(encoding='utf-8')
    source_portal=(BASE/'views/portal.js').read_text(encoding='utf-8')
    html=page.read_text(encoding='utf-8')
    js=portal.read_text(encoding='utf-8')
    if 'id="toolVersions"' not in html:
        start=source_page.index('<section class="panel" id="toolVersions">')
        end=source_page.index('<section class="panel" id="files">',start)
        anchor='<section class="panel" id="files">'
        if anchor not in html: raise ValueError('Neznámý layout staré encyklopedie. Žádné úpravy.')
        html=html.replace(anchor,source_page[start:end]+anchor,1)
        nav='<a href="#phpVersion">PHP verze</a>'
        if nav in html:html=html.replace(nav,nav+'<a href="#toolVersions">Verze jazyků</a>',1)
    if './toolchain-version-report.json' not in js:
        start=source_portal.index('      // Stage 26: read-only evidence')
        end=source_portal.index('      drawArchitecture();makeTree(index.map(f=>f.path));',start)
        anchor='      drawArchitecture();makeTree(index.map(f=>f.path));'
        if anchor not in js:raise ValueError('Neznámý portal.js. Žádné úpravy.')
        js=js.replace(anchor,source_portal[start:end]+anchor,1)
    # All validation complete; no existing graph JSON is rewritten.
    page.write_text(html,encoding='utf-8')
    portal.write_text(js,encoding='utf-8')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description='Offline přehled verzí 7 jazykových/build ekosystémů bez spuštění nástrojů.')
    parser.add_argument('project',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=refresh(args.project,args.output)
    print('Hotovo:',report['summary']['declarations'],'deklarací;',len(report['summary']['languages']),'kategorií.')
    print('Nespustili jsme žádný překladač ani build. Grafové indexy beze změn.')

if __name__=='__main__':main()
