#!/usr/bin/env python3
"""Add PHP version metadata to an existing Atlas site without re-indexing code.

No PHP CLI, Composer, Git, Docker or network calls. Only reads the explicitly
selected project and the site's source-index.json; only writes allowed site files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyzer.php_version_info import inspect_php_versions

BASE=Path(__file__).resolve().parent


def refresh(project: Path, output: Path) -> dict:
    root=project.expanduser().resolve()
    site=output.expanduser().resolve()
    if not root.is_dir() or not site.is_dir() or root==site or site==BASE:
        raise ValueError('Vyžaduje existující odlišnou složku projektu a existující výstup Atlasu.')
    source_index=site/'source-index.json'
    portal=site/'portal.js'
    page=site/'index.html'
    if not all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(site)
               for p in (source_index,portal,page)):
        raise ValueError('Chybí úplný výstup Atlasu nebo obsahuje symbolický odkaz.')
    data=json.loads(source_index.read_text(encoding='utf8'))
    known={f['path'] for f in data.get('files',[]) if isinstance(f,dict) and isinstance(f.get('path'),str)}
    report=inspect_php_versions(root,known)
    dest=site/'php-version-report.json'
    if dest.exists() and (dest.is_symlink() or not dest.resolve().is_relative_to(site)):
        raise ValueError('Odmítnut nebezpečný cíl přehledu verzí.')
    # Preserve the originally generated index, the offline graphics paths and
    # original Atlas UI controls. Add only one self-contained static panel.
    page_text=page.read_text(encoding='utf8')
    portal_text=portal.read_text(encoding='utf8')
    source_page=(BASE/'views/index.html').read_text(encoding='utf8')
    source_portal=(BASE/'views/portal.js').read_text(encoding='utf8')
    if 'id="phpVersion"' not in page_text:
        a=source_page.index('<section class="panel" id="phpVersion">')
        b=source_page.index('<section class="panel" id="files">',a)
        new_panel=source_page[a:b]
        anchor='<section class="panel" id="files">'
        if anchor not in page_text:raise ValueError('Neznámé rozložení starého Atlasu: bez úprav.')
        page_text=page_text.replace(anchor,new_panel+anchor,1)
        if '<a href="#phpVersion">PHP verze</a>' not in page_text:
            nav='<a href="#buildContext">Build kontext</a>'
            if nav in page_text:page_text=page_text.replace(nav,nav+'<a href="#phpVersion">PHP verze</a>',1)
    if "./php-version-report.json" not in portal_text:
        start=source_portal.index('      // PHP version evidence:')
        end=source_portal.index('      drawArchitecture();makeTree(index.map(f=>f.path));',start)
        snippet=source_portal[start:end]
        anchor='      drawArchitecture();makeTree(index.map(f=>f.path));'
        if anchor not in portal_text:raise ValueError('Neznámý portal.js: bez úprav.')
        portal_text=portal_text.replace(anchor,snippet+anchor,1)
    # Validation above must finish before modifying any files. No project data
    # is copied into the site; the resulting JSON contains metadata only.
    page.write_text(page_text,encoding='utf8')
    portal.write_text(portal_text,encoding='utf8')
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description='Offline přehled PHP verzí bez spuštění PHP/Composer a bez nové analýzy grafu.')
    parser.add_argument('project',type=Path,help='Vybraná složka analyzovaného projektu')
    parser.add_argument('--output',type=Path,required=True,help='Existující vygenerovaná složka Atlasu')
    args=parser.parse_args()
    result=refresh(args.project,args.output)
    print(f'Hotovo: {result["php_files_indexed"]} PHP souborů, '
          f'{len(result["evidence"])} verzovaných deklarací, '
          f'{len(result["review_items"])} bodů k revizi.')
    print('PHP/Composer nebyly spuštěny. Datové indexy a grafy zůstaly beze změn.')

if __name__=='__main__':main()
