#!/usr/bin/env python3
"""Repair Stage 20 generated UI without reindexing or reading repository sources.

Only touches two HTML pages inside the explicitly selected Atlas output.  No
network, no scan of the project, no Gradle/Maven or other external process.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from atlas_resource_catalog import html_report

OLD_NOTICE = re.compile(
    r'<aside style="margin:10px;padding:12px;border:1px solid #628;'
    r'border-radius:8px;background:#17222b;color:#fff" role="status">'
    r'(.*?)</aside>',re.S)


def repair(site: Path) -> list[str]:
    site=site.expanduser().resolve()
    if not site.is_dir() or not (site/'index.html').is_file():
        raise ValueError('Atlas output with index.html not found: '+str(site))
    changed=[]
    index=site/'index.html'
    if index.is_symlink() or (site/'resource-manager.html').is_symlink():
        raise ValueError('Symbolically linked UI files cannot be patched')
    body=index.read_text('utf8')
    def replacement(match):
        return ('<div class="atlas-resource-banner" role="status" '
                'style="margin:10px 16px;padding:10px 14px;min-height:0;height:auto;'
                'border:1px solid #435a78;border-radius:9px;background:#172838;'
                'color:#eef7ff;font:13px/1.5 system-ui,Segoe UI,sans-serif">'
                +match.group(1)+'</div>')
    new,n=OLD_NOTICE.subn(replacement,body,count=1)
    if n:
        index.write_text(new,encoding='utf8')
        changed.append('index.html (rozložení opraveno)')
    status=site/'resource-status.json'
    if status.is_file() and not status.is_symlink():
        data=json.loads(status.read_text('utf8'))
        if not isinstance(data,dict) or not isinstance(data.get('graphics'),list):
            raise ValueError('Invalid resource status format')
        mgr=site/'resource-manager.html'
        rendered=html_report(data)
        if not mgr.is_file() or mgr.read_text('utf8')!=rendered:
            mgr.write_text(rendered,encoding='utf8')
            changed.append('resource-manager.html (přímé odkazy na 10 souborů)')
    return changed


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Repair existing Stage 20 generated HTML only; no project scan/network')
    parser.add_argument('--site',type=Path,default=Path(__file__).resolve().parent/'.code-atlas')
    args=parser.parse_args()
    try:
        changes=repair(args.site)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1, f'Chyba opravy: {error}\n')
    print('Atlas UI opraveno. Žádný zdrojový kód se nenačítal ani neposílal.')
    print('Upraveno: '+(', '.join(changes) if changes else 'nic, stránky už byly správně'))
