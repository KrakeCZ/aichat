"""Offline-only dependency catalogue: public upstream references, not a downloader.

No network, no host filesystem discovery. SHA-256 values in installed manifests
prove byte integrity, not upstream authenticity. The catalogue does not infer
that a cached file is an authentic copy of an upstream version.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from prepare_visual_assets import ASSETS
from atlas_resources import GRAPHICS

CATALOG_SCHEMA = 'atlas-public-resource-catalog/1'
NPM = {
    'cytoscape.min.js': 'cytoscape',
    'highlight.min.js': '@highlightjs/cdn-assets',
    'github-dark.min.css': '@highlightjs/cdn-assets',
    'cytoscape-dagre.min.js': 'cytoscape-dagre',
    'layout-base.js': 'layout-base',
    'cose-base.js': 'cose-base',
    'cytoscape-fcose.js': 'cytoscape-fcose',
    'layout-base-v1.js': 'layout-base',
    'avsdf-base.js': 'avsdf-base',
    'cytoscape-avsdf.js': 'cytoscape-avsdf',
    '3d-force-graph.min.js': '3d-force-graph',
    'three-spritetext.min.js': 'three-spritetext',
}
OPTIONAL_ENGINES = [
    {'id':'kotlin-k2', 'title':'JetBrains Kotlin compiler / K2',
     'official':'https://github.com/JetBrains/kotlin/releases',
     'license':'Apache-2.0', 'status':'planned_not_bundled',
     'note':'K2 Analysis API standalone, classpath a verze API dosud nejsou plně integrovány.'},
    {'id':'jdk', 'title':'OpenJDK / javac', 'official':'https://jdk.java.net/',
     'license':'GPL-2.0-with-classpath-exception (distribution-specific)',
     'status':'optional_external', 'note':'Volitelný externí JDK, není součástí Atlasu.'},
    {'id':'llvm-clang', 'title':'LLVM / Clang', 'official':'https://releases.llvm.org/',
     'license':'Apache-2.0 WITH LLVM-exception', 'status':'optional_external',
     'note':'Volitelný místní C/C++ AST backend.'},
    {'id':'golang', 'title':'Go compiler / go/types', 'official':'https://go.dev/dl/',
     'license':'BSD-3-Clause', 'status':'optional_external', 'note':'Volitelný místní Go typový resolver.'},
    {'id':'rust-analyzer', 'title':'rust-analyzer', 'official':'https://github.com/rust-lang/rust-analyzer/releases',
     'license':'MIT OR Apache-2.0', 'status':'planned_not_integrated', 'note':'TODO ve Work, není hotový adaptér.'},
    {'id':'roslyn', 'title':'Microsoft Roslyn', 'official':'https://github.com/dotnet/roslyn',
     'license':'MIT', 'status':'planned_not_integrated', 'note':'TODO ve Work, není hotový adaptér.'},
    {'id':'gradle', 'title':'Gradle', 'official':'https://gradle.org/releases/',
     'license':'Apache-2.0', 'status':'optional_external',
     'note':'Při běžné analýze Atlas nepouští Gradle a nestahuje build závislosti.'},
    {'id':'maven', 'title':'Apache Maven', 'official':'https://maven.apache.org/download.cgi',
     'license':'Apache-2.0', 'status':'optional_external',
     'note':'Při běžné analýze Atlas nepouští Maven.'},
]


def catalog() -> dict:
    from atlas_npm_archive_import import PINNED_FILES, registry_tarball, npm_name
    return {
        'schema': CATALOG_SCHEMA,
        'policy': 'official_links_for_manual_user_action; no automatic download',
        'graphics': [
            {'name':name, 'package':NPM[name], 'version_hint':GRAPHICS[name][0],
             'license_hint':GRAPHICS[name][1],
             'official_package':'https://www.npmjs.com/package/'+quote(NPM[name],safe='')+'/v/'+GRAPHICS[name][0],
             'exact_versioned_asset_url':ASSETS[name],
             'copy_to':'online-resources/inbox/'+name,
             'npm_tarball_url':registry_tarball(NPM[name]),
             'npm_tarball_import':'python atlas_resources.py import-npm --file PACKAGE.tgz --integrity sha512-TRUSTED_BASE64',
             'unverified_origin_notice':'SHA-256 after import verifies stored bytes, not authenticity of the advertised version.'}
            for name in GRAPHICS
        ],
        'engines': OPTIONAL_ENGINES,
    }


def inventory(cache_ok: list[str], cache_bad: list[str], records: dict) -> dict:
    cat = catalog()
    cat['graphics'] = [{**record, 'installed': record['name'] in cache_ok,
                        'invalid': record['name'] in cache_bad,
                        'sha256': records.get(record['name'], {}).get('sha256'),
                        'bytes': records.get(record['name'], {}).get('bytes')}
                       for record in cat['graphics']]
    cat['summary'] = {
        'present': sum(bool(item['installed']) for item in cat['graphics']),
        'required':len(cat['graphics']),
        'graphics_ready':all(item['installed'] for item in cat['graphics']),
        'network_actions':0,
        'host_scan': False,
        'source_code_uploaded':False,
    }
    return cat


def instructions(report: dict, *, missing_only=True) -> str:
    lines = ['Atlas — návod k lokálnímu importu oficiálních prostředků',
             'Při analýze se NIC automaticky nestahuje. Odkazy otevři jen vědomě.',
             'Jména a verze jsou orientační: před použitím ověř vydavatele, licenci a SHA-256.',
             'Kopíruj pouze samostatné .js/.css z uvedených verzovaných distribucí; nic nespouštěj.',
             'Cílová složka: online-resources/inbox/ (uvnitř Atlasu, ne v uživatelském projektu).', '']
    for g in report['graphics']:
        if missing_only and g['installed']:
            continue
        lines += [g['name']+' ['+ ('POŠKOZENO' if g['invalid'] else 'CHYBÍ')+']',
                  '  Oficiální balíček: '+g['official_package'],
                  '  Alternativa: celý npm archiv '+g['npm_tarball_url'],
                  '  Přímý soubor stejné verze: '+g['exact_versioned_asset_url'],
                  '  Ulož jako: '+g['copy_to'],
                  '  Import: python atlas_resources.py import --name '+g['name']+
                  ' --category graphics --file online-resources/inbox/'+g['name']+
                  ' --sha256 <DŮVĚRYHODNÝ_SHA256>', '']
    lines += ['Automatický offline import z předem stažených npm tarballů: python atlas_resources.py npm-packages',
              'Poté: python atlas_resources.py import-npm --file online-resources/inbox/BALIK.tgz --integrity sha512-TRUSTED_BASE64',
              'Npm instalační skripty ani projektový kód se nespouštějí.',
              'Při chybějícím ověřeném hash lze soubor prozatím evidovat lokálně,',
              'ale nelze tvrdit, že pochází od uvedeného vydavatele.',
              'Po všech grafických souborech: python atlas_resources.py activate-graphics',
              'Přenesení přes USB: python atlas_resources.py pack-create --output moje-resources.zip',
              'Na offline PC: python atlas_resources.py pack-install --file moje-resources.zip']
    return '\n'.join(lines) + '\n'


def html_report(report: dict, *, standalone: bool = False) -> str:
    """One-click localhost manager; offline HTML generation never fetches data."""
    from atlas_npm_archive_import import PINNED_FILES, registry_tarball, npm_name
    installed = {item['name'] for item in report['graphics'] if item['installed']}
    rows = []
    for package, (version, assets) in PINNED_FILES.items():
        done = set(assets).issubset(installed)
        title=(npm_name(package) + (' (verze 1.x pro AVSDF)' if package=='layout-base@1.0.2' else ''))
        rows.append(
            '<tr data-pkg="'+html.escape(package,quote=True)+'"><td><b>'+html.escape(title)+'</b>'
            '<small>Verze '+html.escape(version)+' · '+str(len(assets))+' soubor(y)</small></td>'
            '<td class="state">'+('Nainstalováno' if done else 'Chybí')+'</td>'
            '<td><button type="button" class="get-one" data-package="'+html.escape(package,quote=True)+'"'
            +(' disabled' if done else '')+'>'+('Hotovo' if done else 'Stáhnout')+'</button></td>'
            '<td><a href="'+html.escape(registry_tarball(package),quote=True)
            +'" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">Přímý archiv</a></td></tr>'
        )
    engines=''.join('<li>'+html.escape(e['title'])+' · '+html.escape(e['status'])+' · <a href="'
      +html.escape(e['official'],quote=True)+'" rel="noopener noreferrer" target="_blank">web vydavatele</a></li>'
      for e in report['engines'])
    # Offer individual files *even when the complete graphics set is already
    # installed*. The original manager disabled all download buttons at 10/10,
    # leaving no obvious way to save one asset for another offline computer.
    # Only official fixed URLs and cached, known Atlas assets are shown.
    file_rows=[]
    for item in report['graphics']:
        name=item['name']
        safe_name=html.escape(name,quote=True)
        local=('./__resources__/asset/' if standalone else './online-resources/graphics/')+name
        local_action=('<a href="'+html.escape(local,quote=True)+'" download="'+safe_name+'">'
                      'Uložit z Atlasu</a>' if item['installed'] and not item['invalid']
                      else '<span class="muted">Není v cache</span>')
        file_rows.append('<tr><td><code>'+safe_name+'</code></td>'
            '<td>'+('Připraveno' if item['installed'] and not item['invalid'] else 'Chybí')+'</td>'
            '<td>'+local_action+'</td><td><a href="'+html.escape(item['exact_versioned_asset_url'],quote=True)
            +'" target="_blank" rel="noopener noreferrer" referrerpolicy="no-referrer">'
            'Originální soubor</a></td></tr>')
    home=('<span class="muted">Samostatný správce · bez otevřeného projektu</span>' if standalone
          else '<a href="./index.html">Zpět na encyklopedii</a>')
    template=(Path(__file__).resolve().parent/'views'/'resource-manager-template.html').read_text('utf-8')
    return (template.replace('__GRAPHIC_COUNT__',str(report['summary']['present']))
            .replace('__GRAPHIC_MAX__',str(report['summary']['required']))
            .replace('__PACKAGE_ROWS__',''.join(rows))
            .replace('__DIRECT_ASSET_ROWS__',''.join(file_rows))
            .replace('__HOME_LINK__',home)
            .replace('__OPTIONAL_ENGINES__',engines))
