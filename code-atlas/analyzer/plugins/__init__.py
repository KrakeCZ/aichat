"""Autodetected, independently extensible language analyzers.

Each module exports PLUGIN implementing `name`, `languages` and `analyze(context)`.
No plugin executes project source or installs its dependencies.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from analyzer.model import ConfidenceLevel, EdgeEvidence, EdgeType, Graph


@dataclass
class Context:
    root: Path
    graph: Graph
    paths: set[str]
    files_by_language: dict[str, list[str]]
    unresolved: list[dict] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8-sig", errors="replace")

    def add(self, source: str, target: str, line: int, snippet: str, plugin: str,
            edge_type: EdgeType = EdgeType.IMPORT,
            confidence: ConfidenceLevel = ConfidenceLevel.RESOLVED_STATIC) -> None:
        edge = self.graph.add_edge(source, target, edge_type, confidence, plugin,
            EdgeEvidence(source, line, snippet=snippet.strip()[:240]))
        if edge:
            self.counts[plugin] += 1

    def miss(self, plugin: str, source: str, line: int, reference: str) -> None:
        # Diagnostic only, no speculative relationships.
        if len(self.unresolved) < 250:
            self.unresolved.append({"plugin": plugin, "file": source, "line": line, "reference": reference})


class LanguagePlugin(Protocol):
    name: str
    languages: tuple[str, ...]
    def analyze(self, context: Context) -> None: ...


def run_plugins(root: Path, graph: Graph) -> dict:
    # Registering a plugin is one import + one entry in this list.
    from .python_plugin import PLUGIN as python
    from .javascript_plugin import PLUGIN as javascript
    from .powershell_plugin import PLUGIN as powershell
    from .rust_plugin import PLUGIN as rust
    from .go_plugin import PLUGIN as go
    from .shell_plugin import PLUGIN as shell
    from .local_references_plugin import PLUGIN as local_refs

    plugins: list[LanguagePlugin] = [python, javascript, powershell, rust, go, shell, local_refs]
    languages: dict[str, list[str]] = {}
    paths = set()
    for node in graph.nodes.values():
        if node.type.value != "file":
            continue
        paths.add(node.id)
        lang = node.properties.get("language", "unknown")
        languages.setdefault(lang, []).append(node.id)
    context = Context(root, graph, paths, languages)
    active = []
    for plugin in plugins:
        if any(languages.get(lang) for lang in plugin.languages):
            active.append(plugin.name)
            plugin.analyze(context)
    from .declarations import enrich_declarations
    extra = enrich_declarations(context)
    built_in = {"kotlin", "java", "cpp", "c", "header", "xml", "cmake", "gradle_kts"}
    plugin_langs = {lang for plugin in plugins for lang in plugin.languages}
    coverage = {lang: ("vazby + deklarace (částečně)" if lang in built_in or lang in plugin_langs
                       else "soubory + heuristické deklarace" if extra["total_by_language"].get(lang)
                       else "pouze rozpoznání souborů") for lang in languages}
    return {
        "declarations": extra,
        "language_coverage": coverage,
        "detected_languages": {k: len(v) for k, v in sorted(languages.items())},
        "active_plugins": active,
        "edges_by_plugin": dict(Counter(edge.analyzer for edge in graph.edges if edge.analyzer in active)),
        "built_in_analyzers": ["kotlin_java", "native_cpp", "jni", "android_xml", "gradle", "cmake"],
        "unresolved_local_references": context.unresolved,
        "limitations": [
            "Pouze staticky rozpoznatelné vazby; žádná runtime diagnostika.",
            "Dynamické importy, volání a nepřímé závislosti nemusejí být nalezeny.",
            "Externí balíčky se nevykreslují jako interní soubory.",
        ],
    }
