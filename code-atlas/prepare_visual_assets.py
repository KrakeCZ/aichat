#!/usr/bin/env python3
"""Explicit one-time download of the ORIGINAL graphical libraries for offline use.

Never reads/analyzes/sends a project file. Only retrieves public JS/CSS artifacts
at fixed versioned URLs, storing SHA-256 digests for later verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ASSETS = {
    'cytoscape.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.34.3/cytoscape.min.js',
    'highlight.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js',
    'github-dark.min.css': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css',
    'cytoscape-dagre.min.js': 'https://cdn.jsdelivr.net/npm/cytoscape-dagre@4.0.1/dist/cytoscape-dagre.min.js',
    'layout-base.js': 'https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js',
    'cose-base.js': 'https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js',
    'cytoscape-fcose.js': 'https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js',
    'layout-base-v1.js': 'https://cdn.jsdelivr.net/npm/layout-base@1.0.2/layout-base.js',
    'avsdf-base.js': 'https://cdn.jsdelivr.net/npm/avsdf-base@1.0.0/avsdf-base.js',
    'cytoscape-avsdf.js': 'https://cdn.jsdelivr.net/npm/cytoscape-avsdf@1.0.0/cytoscape-avsdf.js',
    '3d-force-graph.min.js': 'https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.1/dist/3d-force-graph.min.js',
    'three-spritetext.min.js': 'https://cdn.jsdelivr.net/npm/three-spritetext@1.10.0/dist/three-spritetext.min.js',
}

def main():
    ap=argparse.ArgumentParser(description='Jednorázově stáhne knihovny z jejich verzovaných vydání, aby Atlas mohl fungovat offline.')
    ap.add_argument('--verify', action='store_true', help='Jen ověřit dříve uložené soubory; žádná síť')
    ap.add_argument('--allow-network', action='store_true', help='Výslovné povolení pro jednorázové stažení všech 10 veřejných grafických artefaktů')
    args=ap.parse_args()
    dest=Path(__file__).resolve().parent/'vendor';dest.mkdir(exist_ok=True)
    mf=dest/'manifest.json'
    if args.verify:
        if not mf.is_file(): raise SystemExit('Nenalezen manifest; nejprve spusť stažení.')
        records=json.loads(mf.read_text(encoding='utf-8'))
        missing=set(ASSETS)-set(records)
        if missing:
            raise SystemExit('Chybí grafické závislosti: '+', '.join(sorted(missing)))
        for name,record in records.items():
            file=dest/name
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:
                raise SystemExit(f'NESEDÍ HASH: {name}')
        print(f'OK: ověřeno {len(records)} vendored assetů, bez síťové komunikace.')
        return
    if not args.allow_network:
        raise SystemExit('Síť je vypnutá. Preferuj `python atlas_resources.py instructions`, nebo výslovně přidej --allow-network. Bez souhlasu NIC nestahuji.')
    records={}
    for name,url in ASSETS.items():
        file=dest/name
        if not file.is_file() or file.stat().st_size < 150:
            print('Stahuji',name,flush=True)
            with urlopen(Request(url,headers={'User-Agent':'Universal-Code-Atlas/OriginPlus'}),timeout=25) as response:
                body=response.read(8_000_001)
            if len(body)<150 or len(body)>8_000_000 or b'<html' in body[:250].lower():
                raise SystemExit(f'Chybný obsah pro {name}; nic neinstaluji.\n{url}')
            file.write_bytes(body)
        records[name]={'url':url,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
    mf.write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Hotovo. Další generování Atlasu může používat lokální grafické knihovny.')
    print('Žádný zdrojový soubor analyzovaného projektu se během tohoto kroku nečte.')

if __name__=='__main__': main()
