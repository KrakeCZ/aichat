#!/usr/bin/env python3
"""Code Atlas Origin+ — original V0 user experience with multi-project analyzers.

Usage:
    python atlas.py run /path/to/project
    python atlas.py analyze /path/to/project --output ./atlas-site
    python atlas.py serve /path/to/project --output ./atlas-site
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import webbrowser
from urllib.parse import unquote, urlsplit

from analyzer.inventory import Inventory
from analyzer.model import ConfidenceLevel, EdgeEvidence, EdgeType
from analyzer.resolver import ArchitectureResolver
from analyzer.validation import GraphValidator
from analyzer.plugins import run_plugins
from analyzer.architecture_summary import summarize_architecture
from analyzer.symbol_graph import build_symbol_graphs
from analyzer.dependency_topology import derive_dependency_topology
from analyzer.language_capabilities import capabilities

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_DIR / ".code-atlas"
LANGUAGE_LABELS = {"kotlin":"Kotlin", "java":"Java", "cpp":"C++", "c":"C", "header":"C/C++ header", "python":"Python", "javascript":"JavaScript", "typescript":"TypeScript", "rust":"Rust", "go":"Go", "xml":"XML", "cmake":"CMake", "gradle_kts":"Gradle/Kotlin", "rust_manifest":"Cargo manifest", "go_manifest":"Go module", "js_config":"JS/TS manifest", "python_manifest":"Python manifest", "shell":"Shell", "powershell":"PowerShell"}


def git_repository(root: Path) -> str:
    """Read public GitHub slug if remote origin is on github.com; never push."""
    try:
        remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=root, stderr=subprocess.DEVNULL, timeout=3).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"(?:github\.com[:/])([^/\s]+)/([^/\s]+?)(?:\.git)?$", remote)
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def _available_local_graphics(base: Path | None = None) -> tuple[str, dict]:
    """Select a complete user-imported offline set or existing verified vendor.

    Never fetches data or recursively searches the machine. Incomplete caches
    cannot be silently treated as complete 2D/3D installations.
    """
    from prepare_visual_assets import ASSETS
    base = base or APP_DIR
    cached = base / 'online-resources'
    if cached.is_symlink():
        raise ValueError('Offline resources must stay inside the Atlas installation')
    meta_file = cached / 'manifest.json'
    if meta_file.is_file():
        resource_manifest = json.loads(meta_file.read_text(encoding='utf-8'))
        if resource_manifest.get('schema') == 'atlas-offline-resources/1':
            records = resource_manifest.get('records', {})
            if all(name in records and records[name].get('category') == 'graphics'
                   for name in ASSETS):
                entries = {}
                for name in ASSETS:
                    sha = records[name]['sha256']
                    if not re.fullmatch(r'[a-f0-9]{64}', sha):
                        raise ValueError('Offline resource manifest has an invalid hash')
                    path = cached / 'blobs' / sha[:2] / sha
                    if path.is_symlink() or not path.resolve().is_relative_to(cached.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                        raise ValueError('Offline graphics cache failed verification: '+name)
                    entries[name] = {'path': path, 'sha256': sha, 'url': ASSETS[name]}
                return 'online-resources/graphics', entries
    vendor = base / 'vendor'
    if vendor.is_symlink():
        raise ValueError('Vendor graphics must stay inside the Atlas installation')
    manifest_file = vendor / 'manifest.json'
    if manifest_file.is_file():
        old = json.loads(manifest_file.read_text(encoding='utf-8'))
        entries = {}
        for name in ASSETS:
            record = old.get(name)
            if not isinstance(record, dict):
                continue
            file = vendor / name
            if file.is_symlink() or not file.is_file() or record.get('url') != ASSETS[name] or not re.fullmatch(r'[0-9a-f]{64}',record.get('sha256','')):
                raise ValueError('Invalid legacy graphics asset: '+name)
            if hashlib.sha256(file.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Graphics checksum mismatch: '+name)
            entries[name] = {'path': file, 'sha256': record['sha256'], 'url': ASSETS[name]}
        return 'vendor', entries
    return '', {}


def create_index(root: Path, prefix: str, repository: str, source_commit: str, graph) -> dict:
    """Preserve the original V0 source-index symbols/imports fields."""
    inv = Inventory(root, prefix)
    files = []
    imports_by_path = defaultdict(set)
    for edge in graph.edges:
        if (edge.type.value if hasattr(edge.type, 'value') else str(edge.type)) in ('import', 'include', 'script_source'):
            imports_by_path[edge.source].add(edge.target)
    for f in inv.scan():
        path = root / f.path
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        node = graph.nodes.get(f.path)
        symbols = [s.get("name") for s in (node.symbols if node else []) if isinstance(s, dict) and s.get("name")]
        if node:
            symbols += [s for s in node.search_symbols if s not in symbols]
        files.append({"path":f.path,"bytes":f.size_bytes,"lines":len(src.splitlines()),
            "language":f.language,"category":f.category,"module":f.module,
            "symbols":list(dict.fromkeys(symbols)),
            "imports":sorted(imports_by_path.get(f.path, set()))})
    return {"schema_version":1,"repository":repository,"source_commit":source_commit,
            "files":sorted(files,key=lambda f:f["path"])}


def coverage_report(graph, report, diagnostics) -> str:
    """Universal successor to original detailed GRAPH_COVERAGE_REPORT.md."""
    from collections import Counter
    types=Counter(str(e.type.value if hasattr(e.type,'value') else e.type) for e in graph.edges)
    conf=Counter(str(e.confidence.value if hasattr(e.confidence,'value') else e.confidence) for e in graph.edges)
    by_analyzer=Counter(e.analyzer for e in graph.edges)
    degree=Counter()
    for e in graph.edges:
        degree[e.source]+=1;degree[e.target]+=1
    isolated=[p for p,n in graph.nodes.items() if n.type.value=='file' and not degree[p]]
    lines=['# GRAPH COVERAGE REPORT — Code Atlas Origin+', '',
       'Statická analýza, ne běhový důkaz. Nulový počet hran není důkaz nezávislosti.', '',
       '## Souhrn', '',f'- Uzly: **{len(graph.nodes)}**',f'- Hrany: **{len(graph.edges)}**',
       f'- Izolované indexované soubory: **{len(isolated)}**',
       f'- Integrita koncových bodů a řádků: **{"OK" if report.is_valid else "PROBLÉM"}**',
       '', '## Typy vazeb', '']
    lines += [f'- `{name}`: {n}' for name,n in sorted(types.items())]
    lines += ['', '## Jistota / úroveň ověření', '']
    lines += [f'- `{name}`: {n}' for name,n in sorted(conf.items())]
    lines += ['', '## Analyzátory', '']
    lines += [f'- `{name}`: {n} hran' for name,n in sorted(by_analyzer.items())]
    lines += ['', '## Pokrytí jazyků', '']
    lines += [f'- `{name}`: {label}' for name,label in sorted(diagnostics.get('language_coverage',{}).items())]
    lines += ['', '## Nevyřešené lokální odkazy (prvních 80)', '']
    lines += [f'- `{item.get("file")}`:{item.get("line")} — `{item.get("reference")}`' for item in diagnostics.get('unresolved_local_references',[])[:80]]
    lines += ['',f'## Izolované soubory (prvních 120 / {len(isolated)})','']
    lines += [f'- `{p}`' for p in isolated[:120]]
    lines += ['', '## Otevřená omezení', '',
      '- Kontrola integrity ověřuje rozsahy důkazů, nikoli pravdivost každého importu nebo volání.',
      '- Vztahy v komentářích a řetězcích nesmějí vytvářet přesné hrany; JS/TS dostal konzervativní lexer.',
      '- Dynamické načítání, reflexe a runtime události mohou být mimo statický graf.',
      '- Funkční závislosti mezi procesy, službami a vlákny vyžadují specializované analyzátory nebo runtime záznam.','']
    return '\n'.join(lines)


def overview(root: Path, graph, report) -> str:
    langs = Counter(n.properties.get("language", "unknown") for n in graph.nodes.values() if n.type.value == "file")
    out_degree, in_degree = Counter(), Counter()
    for e in graph.edges:
        out_degree[e.source] += 1
        in_degree[e.target] += 1
    top = sorted((n for n in graph.nodes if graph.nodes[n].type.value == "file"), key=lambda n: (-(in_degree[n] + out_degree[n]), n))[:12]
    lines = [f"# Atlas projektu: {root.name}", "", "Automaticky vytvořený **statický** přehled. Nejde o runtime trasování ani potvrzení chyby.", "", "## Rozsah", "", f"- Souborové a virtuální uzly: **{len(graph.nodes)}**", f"- Zaznamenané vazby: **{len(graph.edges)}**", f"- Izolované uzly v grafu: **{report.isolated_nodes}**", "", "## Jazyky", ""]
    lines.extend(f"- {LANGUAGE_LABELS.get(lang, lang)}: {count} souborů" for lang, count in langs.most_common())
    lines.extend(["", "## Pokrytí analýzy", ""])
    lines.extend(f"- {LANGUAGE_LABELS.get(lang, lang)}: {coverage}" for lang, coverage in sorted(getattr(graph, "metadata", {}).get("language_coverage", {}).items()))
    lines.extend(["", "## Nejvíce propojené soubory (orientační místa k průzkumu)", ""])
    lines.extend(f"- `{name}` — {in_degree[name]} příchozích, {out_degree[name]} odchozích vazeb" for name in top)
    lines.extend(["", "## Poznámky", "", "- Vazby jsou určeny statickou analýzou; ne všechny jazyky mají úplné pokrytí.", "- Izolovaný soubor nemusí znamenat problém.", "- Bez logů, testů a běhu aplikace nelze určit skutečnou příčinu pádu.", "- Soubor GRAPH_INTEGRITY.json obsahuje kontrolu důkazů, chyb a orphan vazeb."])
    return "\n".join(lines) + "\n"


def analyze(project: Path, output: Path, repository: str, prefix: str = "", bundle_sources: bool = False,
            allow_cdn: bool = False, clojure_aliases=(), clojure_reader_jar_dir=None, clojure_classpath_snapshot=None, clojure_compiler_report=None, clojure_target_jvm=None,
            clojure_compiler_trust_key=None, clojure_compiler_report_signature=None) -> dict:
    from analyzer.semantic_adapters.clojure_source_context import normalize_cli_aliases
    clojure_aliases=normalize_cli_aliases(clojure_aliases)
    if clojure_target_jvm is not None and (type(clojure_target_jvm) is not int or not 8 <= clojure_target_jvm <= 27):
        raise ValueError('--clojure-target-jvm vyžaduje celé číslo 8 až 27')
    if bool(clojure_compiler_trust_key) != bool(clojure_compiler_report_signature):
        raise ValueError('Podepsaný report vyžaduje obě volby --clojure-compiler-trust-key a --clojure-compiler-report-signature')
    if (clojure_compiler_trust_key or clojure_compiler_report_signature) and not clojure_compiler_report:
        raise ValueError('Podepsaný report vyžaduje --clojure-compiler-report')
    root = project.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Složka projektu neexistuje: {root}")
    output = output.expanduser().resolve()
    # Don't overwrite a project tree or the installed toolkit.
    if root == output or output == APP_DIR:
        raise ValueError("Výstupní adresář nesmí být kořen projektu ani složka nástroje.")
    # Never overwrite arbitrary user documents stored in a directory inside
    # the analyzed repository. Reuse is allowed only for an identified Atlas
    # export; a new/empty output directory is harmless and supported.
    if output.is_relative_to(root) and output.exists():
        from analyzer.inventory import is_generated_atlas_export
        if (output.is_dir() and any(output.iterdir()) and
                not is_generated_atlas_export(output)):
            raise ValueError('Výstup uvnitř projektu obsahuje cizí data; '
                             'nelze je přepsat jako Atlas export.')
    if repository and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("--repo očekává GitHub slug ve tvaru vlastnik/repozitar")
    repository = repository or git_repository(root)
    print(f"[1/4] Analyzuji projekt: {root.name} (prefix={prefix!r})", flush=True)
    resolver = ArchitectureResolver(root_dir=root, prefix=prefix, repository=repository)
    graph = resolver.build_graph()
    from analyzer.build_context import analyze_build_context
    build_context = analyze_build_context(root, graph)
    diagnostics = run_plugins(root, graph)
    graph.metadata.update({"project_name": root.name, "scope_prefix": prefix, "origin_ui_engine": "adapted_v0",
                           "source_mode": "bundled" if bundle_sources else "local",
                           "detected_languages": diagnostics["detected_languages"],
                           "active_plugins": diagnostics["active_plugins"],
                           "language_coverage": diagnostics["language_coverage"]})
    print(f"[2/4] Detekováno: {', '.join(diagnostics['active_plugins']) or 'pouze zabudované analyzátory'}; "
          f"{len(graph.nodes)} uzlů / {len(graph.edges)} vazeb", flush=True)
    report = GraphValidator(root).validate(graph)
    data = graph.to_dict()
    data["project_name"] = root.name
    output.mkdir(parents=True, exist_ok=True)
    # Stage 25: PHP VERSION EVIDENCE ONLY. Metadata read within this repository;
    # no PHP / Composer / Docker invocation, network request or disk search.
    from analyzer.php_version_info import inspect_php_versions
    php_versions = inspect_php_versions(root, set(graph.nodes))
    (output / 'php-version-report.json').write_text(
        json.dumps(php_versions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Stage 26: declared toolchains/targets only. No executable lookup or run.
    from analyzer.toolchain_versions import inspect_toolchain_versions
    tool_versions = inspect_toolchain_versions(root, set(graph.nodes))
    (output / 'toolchain-version-report.json').write_text(
        json.dumps(tool_versions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    viewfiles=("index.html","portal.js","atlas-v2.html","graph.html","atlas-view-modes.js","atlas-enhancements.js","atlas-inline-drilldown.js","atlas-node-menu.js","inside-file.html","inside-file.js","thanks.html","timeline.html","timeline.js")
    from prepare_visual_assets import ASSETS
    from atlas_resources import verify as verify_offline_store
    from atlas_resource_catalog import inventory as resource_inventory, html_report as resource_page
    graphics_dir, local_graphics = _available_local_graphics()
    graphics_dir = graphics_dir or 'online-resources/graphics'
    cache_ok, cache_bad, cache_data = verify_offline_store()
    resource_status = resource_inventory(cache_ok, cache_bad, cache_data['records'])
    # Legacy vendor resources also count as locally ready, even before migration.
    resource_status['summary']['renderer_assets_ready'] = len(local_graphics) == len(ASSETS)
    resource_status['summary']['output_policy'] = 'manual_cdn_opt_in' if allow_cdn else 'offline_default'
    resource_status['summary']['missing_renderer_assets'] = sorted(set(ASSETS)-set(local_graphics))
    (output / 'resource-status.json').write_text(
        json.dumps(resource_status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output / 'resource-manager.html').write_text(resource_page(resource_status),encoding='utf8')
    for name, info in local_graphics.items():
        target = output / graphics_dir / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(info['path'], target)
    # Licenses imported from pinned public npm tarballs stay with the generated
    # offline site and with USB resource packs. Never scan the user's project.
    from atlas_npm_archive_import import PINNED_FILES
    license_root = APP_DIR / 'online-resources' / 'licenses'
    license_report = []
    if license_root.is_dir() and not license_root.is_symlink():
        for package, (version, assets) in PINNED_FILES.items():
            # Include a package notice only if all of its files were copied.
            if not set(assets).issubset(local_graphics):
                continue
            safe_name = re.sub(r'[^A-Za-z0-9_.@-]', '_', f'{package}@{version}') + '.LICENSE.txt'
            source_notice = license_root / safe_name
            if not source_notice.is_file() or source_notice.is_symlink() or source_notice.stat().st_size > 300_000:
                continue
            destination_notice = output / 'online-resources' / 'licenses' / safe_name
            destination_notice.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source_notice,destination_notice)
            license_report.append({'package':package,'version':version, 'file':str(destination_notice.relative_to(output)).replace('\\','/'),
                                   'sha256':hashlib.sha256(destination_notice.read_bytes()).hexdigest()})
    if license_report:
        (output / 'online-resources' / 'licenses' / 'manifest.json').write_text(
            json.dumps({'schema':'atlas-upstream-license-notices/1','notices':license_report},
                       indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    resource_status['summary']['included_license_notices'] = len(license_report)
    (output / 'resource-status.json').write_text(
        json.dumps(resource_status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if len(local_graphics) == len(ASSETS):
        from atlas_resources import GRAPHICS
        site_manifest = {
            'schema': 'atlas-offline-site-assets/1',
            'network': 'no_automatic_graphics_requests',
            'files': {
                name: {'sha256':info['sha256'], 'upstream_url':ASSETS[name],
                       'version_hint':GRAPHICS[name][0], 'license_hint':GRAPHICS[name][1]}
                for name, info in sorted(local_graphics.items())
            },
            'notice': 'Local pinned content; SHA-256 cannot prove upstream authenticity by itself.',
        }
        (output/graphics_dir/'manifest.json').write_text(
            json.dumps(site_manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    ready = len(local_graphics) == len(ASSETS)
    for fname in viewfiles:
        body=(APP_DIR / "views" / fname).read_text(encoding="utf-8")
        for filename, url in ASSETS.items():
            # Offline DEFAULT: NEVER leave a CDN asset reference in generated JS.
            # Explicit --allow-cdn restores ONLY the historical browser fallback.
            if not allow_cdn or filename in local_graphics:
                body=body.replace(url, './'+graphics_dir+'/'+filename)
        if fname == 'index.html':
            label = ('Offline grafika připravena' if ready else
                'Offline grafika zatím chybí — zdrojová analýza funguje; pro 2D/3D přidej lokální knihovny')
            # Do NOT use <aside> here: the encyclopedia's global CSS styles
            # every aside as a sticky, 100vh navigation sidebar.  In Stage 20
            # that made this simple message occupy a full screen and displaced
            # the real layout.  A unique non-layout tag preserves original UI.
            banner=('<div class="atlas-resource-banner" role="status" '
                    'style="margin:10px 16px;padding:10px 14px;min-height:0;height:auto;'
                    'border:1px solid #435a78;border-radius:9px;background:#172838;'
                    'color:#eef7ff;font:13px/1.5 system-ui,Segoe UI,sans-serif">'
                    + label + ' · <a style="color:#9cecff" href="./resource-manager.html">Správa prostředků</a>'
                    + ' · <a style="color:#9cecff" href="./ANALYSIS_ASSURANCE.json">Kvalita analýzy: neověřená úplnost</a>'
                    + (' · CDN výslovně povoleno' if allow_cdn else ' · bez automatických síťových požadavků')
                    + '</div>')
            body=body.replace('<body>', '<body>'+banner,1)
        (output/fname).write_text(body,encoding="utf-8")
    if ready:
        print('Grafika: 10 ověřených lokálních prostředků, bez CDN.',flush=True)
    elif allow_cdn:
        print('POZOR: --allow-cdn výslovně dovoluje browseru načíst chybějící grafické knihovny ze sítě.',flush=True)
    else:
        print('OFFLINE DEFAULT: chybějí místní grafické knihovny. Žádné CDN;'
              ' analyzovaná data zůstanou dostupná. Otevři resource-manager.html.',flush=True)
    # Project-neutral exports must NEVER include an unrelated example project's
    # source tree or historical atlas graph. Original V0 is an external reference.
    if bundle_sources:
        sources = output / "sources"
        for file_id in sorted(n.id for n in graph.nodes.values() if n.type.value == "file"):
            original = (root / file_id).resolve()
            if not original.is_file() or not original.is_relative_to(root) or original.stat().st_size > 2_000_000:
                continue
            target = sources / file_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original,target)
    index = create_index(root, prefix, repository, graph.source_commit, graph)
    data['_atlas_clojure_cli_aliases']=list(clojure_aliases)
    if clojure_target_jvm is not None:data['_atlas_clojure_target_jvm']=clojure_target_jvm
    if clojure_reader_jar_dir: data['_atlas_clojure_reader_jar_dir']=str(Path(clojure_reader_jar_dir).expanduser().resolve())
    if clojure_classpath_snapshot: data['_atlas_clojure_classpath_snapshot']=str(Path(clojure_classpath_snapshot).expanduser().resolve())
    if clojure_compiler_report: data['_atlas_clojure_compiler_report']=str(Path(clojure_compiler_report).expanduser())
    if clojure_compiler_trust_key: data['_atlas_clojure_compiler_trust_key']=str(Path(clojure_compiler_trust_key).expanduser())
    if clojure_compiler_report_signature: data['_atlas_clojure_compiler_report_signature']=str(Path(clojure_compiler_report_signature).expanduser())
    try:
        symbol_catalogue = build_symbol_graphs(root, index, data, output)
    finally:
        # Internal input to language adapter, NOT part of the graph-index model.
        data.pop('_atlas_clojure_cli_aliases',None)
        data.pop('_atlas_clojure_target_jvm',None)
        data.pop('_atlas_clojure_reader_jar_dir',None)
        data.pop('_atlas_clojure_classpath_snapshot',None)
        data.pop('_atlas_clojure_compiler_report',None)
        data.pop('_atlas_clojure_compiler_trust_key',None)
        data.pop('_atlas_clojure_compiler_report_signature',None)
    promoted = symbol_catalogue.pop('_promoted_edges', [])
    for link in promoted:
        relation = link['relation']
        kind = (EdgeType.SYMBOL_CALL if relation == 'calls' else
                EdgeType.SYMBOL_INSTANTIATION if relation == 'instantiates' else EdgeType.INHERITANCE)
        ev = link['evidence']
        graph.add_edge(link['source'], link['target'], kind, ConfidenceLevel.RESOLVED_STATIC,
                       link['analyzer'], EdgeEvidence(ev['path'], int(ev['line']),
                           int(ev.get('column',0)), ev.get('snippet','')))
    # Optional candidate relations from per-language syntax and JNI ABI naming.
    # These must NEVER appear as exact static bindings in the architecture.
    candidates = symbol_catalogue.pop('_candidate_symbol_edges', [])
    for link in candidates:
        kind = (EdgeType.JNI_BINDING if link['relation'] in ('jni_name_match','jni_register_natives')
                else EdgeType.INCLUDE if link['relation'] == 'include_candidate'
                else EdgeType.SQL_TABLE_REFERENCE if link['relation'] == 'sql_table_candidate'
                else EdgeType.CLOJURE_REQUIRE if link['relation'] == 'clojure_require_candidate'
                else EdgeType.CLOJURE_MACRO if link['relation'] in ('clojure_macro_invocation_candidate','clojure_macro_phase_candidate','clojure_referred_macro_invocation_candidate','clojure_referred_macro_phase_candidate')
                else EdgeType.CALL_CANDIDATE)
        ev = link['evidence']
        graph.add_edge(link['source'], link['target'], kind, ConfidenceLevel.CANDIDATE,
                       link['analyzer'], EdgeEvidence(ev['path'], int(ev['line']),
                           int(ev.get('column',0)), ev.get('snippet','')))
    graph.metadata['candidate_symbol_cross_file_evidence'] = len(candidates)
    graph.metadata['resolved_symbol_cross_file_evidence'] = len(promoted)
    # Stage 16: annotate possible Android source-set/variant overlap only
    # AFTER all symbol links exist. This does not manufacture or delete edges.
    from analyzer.android_variants import analyze_android_variants
    android_variants = analyze_android_variants(root, graph)
    (output / "android-variant-context.json").write_text(
        json.dumps(android_variants, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph.metadata['android_variant_model'] = 'literal_candidates_not_resolved_AGP'
    graph.metadata['android_variant_disjoint_edges'] = len(android_variants['edge_annotations'])
    report = GraphValidator(root).validate(graph)
    data = graph.to_dict()
    data['project_name'] = root.name
    (output / "graph-index.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    architecture = summarize_architecture(graph)
    (output / "build-context.json").write_text(json.dumps(build_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "architecture-summary.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "source-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    topology = derive_dependency_topology(data)
    (output / "dependency-topology.json").write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "language-capabilities.json").write_text(json.dumps(capabilities(data, symbol_catalogue), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Symbolové grafy: {symbol_catalogue['coverage']}; aktivní sémantické adaptéry: {symbol_catalogue['semantic_adapters']}", flush=True)
    integrity = {"valid": report.is_valid, "nodes": report.total_nodes, "edges": report.total_edges, "isolated_nodes": report.isolated_nodes, "orphan_edges": report.orphan_edges[:60], "invalid_evidence": report.invalid_evidence[:60], "note": "Project-neutral schema and evidence validation; no application-specific regression gate."}
    (output / "ANALYSIS_REPORT.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "GRAPH_INTEGRITY.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Explicit analytical quality gate: a structurally valid graph MUST NOT
    # silently become a claim of complete, forensic-grade semantic correctness.
    # This only inspects the already-produced local metadata.
    from analyzer.analysis_assurance import build_assurance
    symbol_integrity = json.loads((output / "SYMBOL_INTEGRITY.json").read_text(encoding="utf-8"))
    assurance = build_assurance(index, symbol_catalogue, integrity, symbol_integrity, data)
    (output / "ANALYSIS_ASSURANCE.json").write_text(
        json.dumps(assurance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not assurance['candidate_classification_passed']:
        # Do not leave a misleading global graph that claims candidate relations
        # are proven. Keep the diagnostic report, invalidate the graph and fail.
        (output / "graph-index.json").unlink(missing_ok=True)
        raise RuntimeError("Analysis evidence breach: candidate relation promoted to resolved. "
                           "Review ANALYSIS_ASSURANCE.json; no final graph published.")
    (output / "ARCHITECTURE_OVERVIEW.md").write_text(overview(root, graph, report), encoding="utf-8")
    coverage_text = coverage_report(graph, report, diagnostics)
    coverage_text += ("\n\n## Analytická úplnost – necertifikováno\n\n"
                      "Strukturní integrita a SHA-256 **neprokazují správnost všech jazykových"
                      " konstrukcí, volání ani běhového toku**. Výstup není certifikován"
                      " pro úplné sémantické/forenzní rozhodování."
                      " Kandidátní vazby nejsou potvrzené."
                      " Viz `ANALYSIS_ASSURANCE.json`.\n")
    (output / "GRAPH_COVERAGE_REPORT.md").write_text(coverage_text, encoding="utf-8")
    print(f"[3/4] Výstup: {output}", flush=True)
    if not report.is_valid:
        print(f"POZOR: {len(report.orphan_edges)} orphan vazeb, {len(report.invalid_evidence)} problematických důkazů; podrobnosti v GRAPH_INTEGRITY.json", file=sys.stderr)
    print(f"[4/4] Hotovo: {len(graph.nodes)} uzlů / {len(graph.edges)} vazeb. {len(index['files'])} souborů v indexu.", flush=True)
    print("Do výstupu se kopírují pouze aktuální projektová data; žádný cizí archiv ani historický graf.")
    return {"nodes": len(graph.nodes), "edges": len(graph.edges), "valid": report.is_valid, "repository": repository, "plugins": diagnostics["active_plugins"]}


class SourceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, project: Path, site: Path, allowed_paths: frozenset[str],
                 source_mode: str = 'local', source_commit: str = '', source_hashes: dict | None = None,
                 **kwargs):
        self.project = project
        self.site = site
        self.allowed_paths = allowed_paths
        self.source_mode = source_mode
        self.source_commit = source_commit
        self.source_hashes = source_hashes or {}
        super().__init__(*args, directory=str(site), **kwargs)

    def list_directory(self, path):
        self.send_error(404, "Directory listing disabled")
        return None

    def do_GET(self):
        # Resource API reads only Atlas's own cache; never touches project sources
        # and never starts a download in response to GET/page loading.
        from atlas_resource_http import get as atlas_resource_get
        if atlas_resource_get(self):
            return
        url = urlsplit(self.path)
        if url.path.startswith("/__source__/"):
            path = unquote(url.path.removeprefix("/__source__/"))
            if path.startswith(("/", "\\")) or "\\" in path or "\x00" in path or path not in self.allowed_paths:
                self.send_error(403, "Source not indexed or invalid path")
                return
            if self.source_mode == 'git_snapshot':
                # Read the exact analyzed Git blob, even if the checkout changed
                # after indexing. Never execute any project code.
                if not re.fullmatch(r'[0-9a-f]{40,64}', self.source_commit):
                    self.send_error(500, "Missing exact historical revision")
                    return
                try:
                    result = subprocess.run(['git','-C',str(self.project),'show',
                                            self.source_commit+':'+path],stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,timeout=12,check=False)
                except (OSError, subprocess.SubprocessError):
                    self.send_error(500, "Historical source unavailable")
                    return
                if result.returncode:
                    self.send_error(404, "File is not present in selected Git revision")
                    return
                content = result.stdout
                expected = self.source_hashes.get(path)
                if expected and hashlib.sha256(content).hexdigest() != expected:
                    self.send_error(409, "Git source digest differs from analyzed snapshot")
                    return
            else:
                # Source navigation is only to the *indexed* physical input.
                # Reject symlink swaps, including links whose destination is
                # within the project. This still does not claim atomic OS
                # containment against concurrent hostile filesystem mutation.
                lexical=self.project/path
                current=lexical
                while current!=self.project:
                    if current.is_symlink():
                        self.send_error(403, "Source path changed into symlink")
                        return
                    current=current.parent
                source=lexical.resolve()
                if not source.is_relative_to(self.project) or not source.is_file():
                    self.send_error(403, "Outside project or missing file")
                    return
                if source.stat().st_size > 2_000_000:
                    self.send_error(413, "Source file too large")
                    return
                content=source.read_bytes()
                # Prevent serving a modified source under the old symbol AST
                # and SHA evidence. A new analysis must be run on the change.
                expected=self.source_hashes.get(path)
                if expected and hashlib.sha256(content).hexdigest()!=expected:
                    self.send_error(409, "Source changed since analysis")
                    return
            if len(content) > 3_000_000:
                self.send_error(413, "Source file too large")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        super().do_GET()

    def do_POST(self):
        # Only a CSRF-protected user click at the local resource manager may
        # initiate a download. No other POST endpoint is implemented.
        from atlas_resource_http import post as atlas_resource_post
        if not atlas_resource_post(self):
            self.send_error(404, 'Unknown local resource action')


def serve(project: Path, output: Path, port: int, open_browser: bool = False, start_page: str = "index.html") -> None:
    root = project.expanduser().resolve()
    site = output.expanduser().resolve()
    if not (site / "index.html").is_file():
        raise ValueError(f"Nejdřív spusť analyze (chybí {site / 'index.html'}).")
    # The full indexed file tree is browsable, including files without graph edges.
    index = json.loads((site / "source-index.json").read_text(encoding="utf-8"))
    allowed_paths = frozenset(f["path"] for f in index.get("files", []) if f.get("path"))
    source_mode = index.get('source_mode', 'local')
    source_hashes = {}
    if (site / 'symbol-index.json').is_file():
        symbol_idx = json.loads((site / 'symbol-index.json').read_text(encoding='utf-8'))
        source_hashes = {p: info.get('source_sha256') for p,info in symbol_idx.get('files',{}).items()}
    handler = functools.partial(SourceHandler, project=root, site=site, allowed_paths=allowed_paths,
                                source_mode=source_mode, source_commit=index.get('source_commit',''),
                                source_hashes=source_hashes)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        from atlas_resource_http import configure_server
        configure_server(httpd)
        bound_port = httpd.server_address[1]
        print(f"Otevři v prohlížeči: http://127.0.0.1:{bound_port}/", flush=True)
        print("Lokální server běží pouze na 127.0.0.1. Ukončíš ho Ctrl+C.", flush=True)
        if open_browser:
            try:
                webbrowser.open(f"http://127.0.0.1:{bound_port}/{start_page}")
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer ukončen.")


def require_offline_graphics(base: Path = APP_DIR) -> dict:
    """Fail before analysis when not all versioned graphics assets are local."""
    from prepare_visual_assets import ASSETS
    location, records = _available_local_graphics(base)
    missing = sorted(set(ASSETS) - set(records))
    if missing:
        raise ValueError('Offline graphics are not installed: ' + ', '.join(missing) +
                         '. Use `python atlas_resources.py status` and import the missing files. '
                         'No download attempted.')
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Universal Code Atlas -- komunitní preview")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for command in ("analyze", "run", "serve", "history", "history-run"):
        parser = sub.add_parser(command)
        parser.add_argument("project", nargs="?", default=".", help="Cesta k projektu (výchozí aktuální složka)")
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Adresář s vygenerovaným atlasem")
        parser.add_argument("--port", type=int, default=0, help="HTTP port pro run/serve (0 = automaticky)")
        parser.add_argument("--repo", default="", help="GitHub repo vlastnik/repo (jinak automaticky z origin)")
        parser.add_argument("--prefix", default="", help="Volitelně omezit sken na podadresář")
        parser.add_argument("--bundle-sources", action="store_true", help="Pro statický GitHub Pages web: přibalit všechny indexované zdrojáky (VEŘEJNÁ DATA)")
        parser.add_argument("--offline-graphics", action="store_true", help="Přísný režim: vyžaduje kompletní lokální grafiku před skenem projektu (CI)")
        parser.add_argument("--allow-cdn", action="store_true", help="Výjimečně povolit prohlížeči chybějící CDN knihovny; bez tohoto přepínače je Atlas offline")
        parser.add_argument("--clojure-aliases", default="", metavar=":test:dev",
            help="Výslovně vybrané Clojure CLI aliasy pro STATICKÝ scénář cest; žádný build, CLI ani externí basis")
        parser.add_argument('--clojure-reader-jar-dir', type=Path, default=None,
            help='VOLITELNÉ: lokální JVM nástroj s pevnými JARy pro reader a vybraná clojure.core makra; projektový kód nikdy nenačítá')
        parser.add_argument('--clojure-classpath-snapshot', type=Path, default=None,
            help='Volitelný lokální JSON classpath snapshot NEBO Clojure CLI .basis soubor; žádný download ani spuštění CLI')
        parser.add_argument('--clojure-target-jvm', type=int, default=None,
            help='Explicitní cílová verze JVM (8 až 27) pro čtení multi-release JAR; není detekcí runtime')
        parser.add_argument('--clojure-compiler-report', type=Path, default=None,
            help='Volitelný externě vytvořený offline JSON report kompilátoru: pouze kontrola shody s existujícími kandidáty; NENÍ autentizací kompilátoru')
        parser.add_argument('--clojure-compiler-trust-key', type=Path, default=None,
            help='Volitelný lokální JSON s veřejným Ed25519 klíčem a očekávaným producer ID; pouze pro report v2')
        parser.add_argument('--clojure-compiler-report-signature', type=Path, default=None,
            help='Volitelný oddělený Ed25519 podpis (Base64) přes přesné bajty reportu v2')
        if command in ('history','history-run'):
            parser.add_argument('--history-limit', type=int, default=8, help='Počet Git revizí ke srovnání, maximálně 60')
            parser.add_argument('--history-stride', type=int, default=1, help='Každý N-tý commit v first-parent historii')
            parser.add_argument('--history-ref', default='HEAD', help='Git revize / větev pro časovou osu (výchozí HEAD)')
    args = ap.parse_args()
    try:
        if args.cmd=='serve' and args.clojure_aliases:
            raise ValueError('--clojure-aliases platí jen při analýze, nikoli pouze serve')
        if args.cmd=='serve' and args.clojure_target_jvm is not None:
            raise ValueError('--clojure-target-jvm platí jen při analýze')
        if args.cmd=='serve' and args.clojure_compiler_report:
            raise ValueError('--clojure-compiler-report platí jen při analýze, nikoli pouze serve')
        if args.cmd=='serve' and (args.clojure_compiler_trust_key or args.clojure_compiler_report_signature):
            raise ValueError('Volby podpisu kompilátorového reportu platí jen při analýze')
        if args.offline_graphics and args.allow_cdn:
            raise ValueError('Nelze současně použít --offline-graphics a --allow-cdn')
        if args.offline_graphics:
            require_offline_graphics()
        if args.cmd in ('history','history-run'):
            if args.clojure_classpath_snapshot:
                raise ValueError('--clojure-classpath-snapshot zatím není podporováno pro historickou analýzu')
            if args.clojure_reader_jar_dir:
                raise ValueError('--clojure-reader-jar-dir zatím není podporováno pro historickou analýzu')
            if args.clojure_compiler_report or args.clojure_compiler_trust_key or args.clojure_compiler_report_signature:
                raise ValueError('Volby kompilátorového reportu nejsou podporované pro historickou analýzu')
            if args.clojure_target_jvm is not None:
                raise ValueError('--clojure-target-jvm není podporováno pro historickou analýzu')
            if args.bundle_sources:
                raise ValueError('Režim history automaticky nepublikuje historické zdrojové soubory. --bundle-sources není podporováno.')
            from analyzer.git_history import make_timeline
            from analyzer.git_history import repository_root
            git_root = repository_root(Path(args.project).expanduser().resolve())
            repo = args.repo or git_repository(git_root)
            result = make_timeline(git_root,args.output,ref=args.history_ref,
                                   limit=args.history_limit,stride=args.history_stride,
                                   repository=repo,prefix=args.prefix,analyze_fn=(lambda p,o,r,prefix,b: analyze(p,o,r,prefix,b,allow_cdn=args.allow_cdn,clojure_aliases=args.clojure_aliases)))
            print(f'Historie: {len(result["commits"])} revizí; otevři {args.output / "timeline.html"}')
            if args.cmd == 'history-run':
                serve(git_root,args.output,args.port,open_browser=True,start_page='timeline.html')
        if args.cmd in ("analyze", "run"):
            analyze(Path(args.project), args.output, args.repo, args.prefix, args.bundle_sources,
                allow_cdn=args.allow_cdn, clojure_aliases=args.clojure_aliases, clojure_reader_jar_dir=args.clojure_reader_jar_dir, clojure_classpath_snapshot=args.clojure_classpath_snapshot, clojure_compiler_report=args.clojure_compiler_report, clojure_target_jvm=args.clojure_target_jvm,
                clojure_compiler_trust_key=args.clojure_compiler_trust_key,
                clojure_compiler_report_signature=args.clojure_compiler_report_signature)
        if args.cmd in ("serve", "run"):
            serve(Path(args.project), args.output, args.port, open_browser=(args.cmd == "run"))
        return 0
    except (ValueError, OSError) as ex:
        print(f"Chyba: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
