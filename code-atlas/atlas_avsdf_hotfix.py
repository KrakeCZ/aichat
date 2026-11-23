#!/usr/bin/env python3
"""Repair only the AVSDF JavaScript loader inside an EXISTING Atlas site.

Does not scan/analyze repository sources, execute project code, or use network.
Unknown/custom renderer files are not overwritten. Existing symbol indexes and
other visual controls remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

APP = Path(__file__).resolve().parent
START = '  async function ensureLayout(name) {'
END = '\n  // Existing Atlas functions call renderGraph() dynamically.'
AVSDF_BLOCK = re.compile(r'(?m)^    avsdf: \{\n.*?^    \}', flags=re.S)
LEGACY_PLUGIN = 'https://cdn.jsdelivr.net/npm/cytoscape-avsdf@1.0.0/cytoscape-avsdf.js'


def repair(site: Path) -> list[str]:
    site = Path(site).expanduser().resolve()
    target = site / 'atlas-view-modes.js'
    if not site.is_dir() or not target.is_file() or target.is_symlink():
        raise ValueError('Existing Atlas site with regular atlas-view-modes.js required')
    original = target.read_text('utf8')
    already_patched = 'let avsdfLegacyLayoutBase = null;' in original
    renderer = (APP/'views'/'atlas-view-modes.js').read_text('utf8')
    old_block = AVSDF_BLOCK.search(original)
    new_block = AVSDF_BLOCK.search(renderer)
    # An already patched renderer still needs its local resource-manager HTML
    # updated from the actual (possibly 10/12) cache state.
    if already_patched:
        return _refresh_manager(site)
    if not old_block or not new_block:
        raise ValueError('AVSDF settings not recognized; renderer left unchanged')
    old_urls = re.findall(r'"([^"]+)"', old_block.group(0))
    # The only URL in the old AVSDF config must be the original pinned asset.
    # Do not silently modify arbitrary user JS or unrecognized online hosts.
    if len(old_urls) < 2:
        raise ValueError('AVSDF settings differ from supported Stage 21')
    urls = [s for s in old_urls if s.endswith('cytoscape-avsdf.js')]
    if len(urls) != 1:
        raise ValueError('AVSDF extension URL ambiguous; renderer left unchanged')
    ext_url = urls[0]
    if ext_url == LEGACY_PLUGIN:
        prefix = 'https://cdn.jsdelivr.net/npm/'
        rendered_block = new_block.group(0)
    else:
        allowed = ('./online-resources/graphics/', './vendor/')
        if not any(ext_url == prefix+'cytoscape-avsdf.js' for prefix in allowed):
            raise ValueError('Unknown AVSDF extension location; renderer left unchanged')
        # A legacy site might have the old plugin under ./vendor/, while the
        # current manager always publishes the two NEW dependencies into its
        # site-local online-resources/graphics/ directory.
        prefix = ext_url.removesuffix('cytoscape-avsdf.js')
        addendum_prefix = ('./online-resources/graphics/' if prefix == './vendor/' else prefix)
        from prepare_visual_assets import ASSETS
        rendered_block = new_block.group(0)
        for name in ('layout-base-v1.js','avsdf-base.js'):
            rendered_block = rendered_block.replace(ASSETS[name], addendum_prefix+name)
        rendered_block = rendered_block.replace(ASSETS['cytoscape-avsdf.js'],ext_url)
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError('Renderer layout function changed; no automatic patch applied')
    if renderer.count(START) != 1 or renderer.count(END) != 1:
        raise ValueError('Installed Atlas renderer is not recognized')
    old_layout = original[original.index(START):original.index(END)]
    if ('for (const url of config.urls)' not in old_layout or
            'cytoscape.use(register)' not in old_layout or
            'layoutLoadQueue' in old_layout):
        raise ValueError('Renderer has custom layout logic; no automatic patch applied')
    new_layout = renderer[renderer.index('  // Never load layout-base 1.x'):renderer.index(END)]
    patched = original[:old_block.start()] + rendered_block + original[old_block.end():]
    patched = patched.replace(old_layout,new_layout,1)
    if patched == original:
        return []
    target.write_text(patched,encoding='utf8')
    return ['atlas-view-modes.js (AVSDF dependencies and safe 1.x/2.x isolation)', *_refresh_manager(site)]


def _refresh_manager(site: Path) -> list[str]:
    """Make the installed manager show 10/12 instead of stale Stage-21 10/10.

    Only reads Atlas's own versioned offline-resource manifest, not project
    files or other system paths. No network or analyzed source-code scan.
    """
    from atlas_resource_catalog import html_report, inventory
    from atlas_resources import verify
    from atlas_npm_archive_import import PINNED_FILES
    from atlas_npm_oneclick import package_ready
    import atlas_resources
    import json

    installed, invalid, saved = verify()
    report = inventory(installed, invalid, saved['records'])
    report['packages'] = [{'name':name,'version':version,
                           'installed':package_ready(atlas_resources,name),
                           'assets':sorted(paths)}
                          for name,(version,paths) in PINNED_FILES.items()]
    changed=[]
    mgr=site/'resource-manager.html'
    if mgr.is_file():
        if mgr.is_symlink():
            raise ValueError('Symlinked resource manager is not supported')
        content=html_report(report)
        if mgr.read_text('utf8')!=content:
            mgr.write_text(content,encoding='utf8')
            changed.append('resource-manager.html (complete pinned library list)')
    status=site/'resource-status.json'
    if status.is_file():
        if status.is_symlink():
            raise ValueError('Symlinked resource status is not supported')
        old=json.loads(status.read_text('utf8'))
        summary=old.get('summary', {})
        if isinstance(summary,dict):
            report['summary']['output_policy']=summary.get('output_policy','offline_default')
            report['summary']['included_license_notices']=summary.get('included_license_notices',0)
        from prepare_visual_assets import ASSETS
        report['summary']['renderer_assets_ready']=len(installed)==len(ASSETS) and not invalid
        report['summary']['missing_renderer_assets']=sorted(set(ASSETS)-set(installed))
        content=json.dumps(report,indent=2,ensure_ascii=False)+'\n'
        if status.read_text('utf8')!=content:
            status.write_text(content,encoding='utf8')
            changed.append('resource-status.json (missing AVSDF dependencies)')
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description='Repair AVSDF in existing generated Atlas; no reindex or network')
    parser.add_argument('--site',type=Path,default=APP/'.code-atlas')
    options = parser.parse_args()
    try:
        updated = repair(options.site)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2,'Oprava AVSDF: '+str(exc)+'\n')
    print('Hotovo: '+(', '.join(updated) if updated else 'AVSDF loader already current'))
    print('Symbolové indexy a zdrojové soubory nebyly čteny ani změněny.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
