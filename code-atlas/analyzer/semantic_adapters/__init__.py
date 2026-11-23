"""Project-neutral, opt-in semantic adapters for symbol-graph generation.

Each adapter receives the same context and returns the same evidence-bearing
per-file graph schema. Adapter selection depends solely on detected languages,
not repository, application, package or author names. No project source is run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol, Any


@dataclass(frozen=True)
class AdapterContext:
    root: Path
    candidates: dict[str, dict]
    graph_data: dict


class Adapter(Protocol):
    name: str
    languages: tuple[str, ...]
    def analyze(self, context: AdapterContext) -> tuple[dict, list[dict]]: ...


@dataclass(frozen=True)
class RegisteredAdapter:
    name: str
    languages: tuple[str, ...]
    impl: Callable[[AdapterContext], tuple[dict, list[dict]]]

    def analyze(self, context: AdapterContext) -> tuple[dict, list[dict]]:
        return self.impl(context)


def _python(ctx):
    import sys
    from analyzer.symbol_graph import PythonFile, _sha, _analyze_python_references
    parseds = {}; warnings = []
    root = ctx.root.resolve()
    for path, item in ctx.candidates.items():
        if item.get('language') != 'python':
            continue
        file = (root / path).resolve()
        if not file.is_file() or not file.is_relative_to(root):
            warnings.append({'path':path, 'reason':'missing_or_outside_root'}); continue
        if file.stat().st_size > 2_000_000:
            warnings.append({'path':path, 'reason':'source_exceeds_2mb'}); continue
        try:
            data=file.read_bytes()
            parseds[path]=PythonFile(path,data.decode('utf-8-sig'),_sha(data))
        except (UnicodeError,OSError,SyntaxError,RecursionError,ValueError,OverflowError) as exc:
            warnings.append({'path':path,'reason':type(exc).__name__,
                             'details':str(exc)[:170], 'semantic_nodes_published':False})
        except Exception as exc:
            # An internal AST indexing error in one file must not discard the
            # entire project. Keep this file OUT of the semantic results; the
            # central exporter must retain file-only evidence, never guess
            # declarations from a partially parsed source.
            warnings.append({'path':path,'reason':'python_ast_file_failed',
                             'details':f'{type(exc).__name__}: {str(exc)[:170]}',
                             'semantic_nodes_published':False})
    # A single malformed or exotic Python file must never make the adapter
    # discard AST evidence for every otherwise valid file in a large project.
    # Resolve project-wide first. On an internal resolver exception, reconstruct
    # pristine syntax graphs before attempting independent per-file resolution:
    # no edges from a partially mutated failed pass may escape as evidence.
    resolution_mode = 'project_wide_ast'
    try:
        _analyze_python_references(parseds)
    except Exception as exc:
        warnings.append({'reason': 'python_project_resolver_failed',
                         'details': f'{type(exc).__name__}: {str(exc)[:220]}',
                         'semantic_nodes_published': True,
                         'resolution_scope': 'per_file_only'})
        resolution_mode = 'per_file_ast_after_project_resolver_failure'
        recovered = {}
        for path, original in parseds.items():
            clean = PythonFile(path, original.source, original.hash)
            try:
                _analyze_python_references({path: clean})
            except Exception as file_exc:
                # The individual pass may already have emitted relationships.
                # Reparse to restore declarations-only evidence atomically.
                clean = PythonFile(path, original.source, original.hash)
                warnings.append({'path': path, 'reason': 'python_file_resolver_failed',
                                 'details': f'{type(file_exc).__name__}: {str(file_exc)[:180]}',
                                 'semantic_nodes_published': True,
                                 'resolution_scope': 'syntax_declarations_only'})
            recovered[path] = clean
        parseds = recovered
    return {p:{'nodes':d.nodes,'edges':d.edges,'parser':'python_ast',
               'resolution_scope': resolution_mode,
               'parser_version':f'CPython {sys.version_info.major}.{sys.version_info.minor}',
               'python_source_syntax':d.python_source_syntax,
               'hash':d.hash,'caution':'Python AST and scoped source declarations; dynamic imports, runtime rebinding and dispatch remain unresolved.'}
            for p,d in parseds.items()}, warnings


def _typescript(ctx):
    from analyzer.typescript_ast import analyze_typescript
    return analyze_typescript(ctx.root, ctx.candidates, ctx.graph_data)


def _powershell(ctx):
    from analyzer.symbol_graph import _powershell_file_graphs
    return _powershell_file_graphs(ctx.root, ctx.candidates, ctx.graph_data)


def _kotlin(ctx):
    from .kotlin_psi import analyze_kotlin
    return analyze_kotlin(ctx.root,ctx.candidates,ctx.graph_data)


def _java(ctx):
    from .java_ast import analyze_java
    return analyze_java(ctx.root, ctx.candidates, ctx.graph_data)


def _clang(ctx):
    from .clang_ast import analyze_clang
    return analyze_clang(ctx.root, ctx.candidates, ctx.graph_data)

def _go(ctx):
    from .go_ast import analyze_go
    return analyze_go(ctx.root, ctx.candidates, ctx.graph_data)

def _swift(ctx):
    from .swift_ast import analyze_swift
    return analyze_swift(ctx.root, ctx.candidates, ctx.graph_data)


def _objc(ctx):
    from .clang_ast import analyze_clang
    return analyze_clang(ctx.root, ctx.candidates, ctx.graph_data, languages=('objective_c',))


def _php(ctx):
    from .php_tokens import analyze_php
    return analyze_php(ctx.root, ctx.candidates, ctx.graph_data)


def _ruby(ctx):
    from .ruby_ripper import analyze_ruby
    return analyze_ruby(ctx.root, ctx.candidates, ctx.graph_data)


def _clojure(ctx):
    from .clojure_forms import analyze_clojure
    return analyze_clojure(ctx.root, ctx.candidates, ctx.graph_data)


def _xml(ctx):
    from .xml_structural import analyze_xml
    return analyze_xml(ctx.root, ctx.candidates, ctx.graph_data)


def _sql(ctx):
    from .sql_structural import analyze_sql
    return analyze_sql(ctx.root, ctx.candidates, ctx.graph_data)


# Native, network-free source-structure readers for previously unindexed
# languages. No declared support here implies full compiler semantics.
def _offline_reader(language: str):
    def analyze(ctx):
        from .offline_multilanguage import analyze_offline_multilanguage
        return analyze_offline_multilanguage(ctx.root, ctx.candidates, ctx.graph_data, (language,))
    return analyze


REGISTRY: tuple[RegisteredAdapter, ...] = (
    RegisteredAdapter('python-ast', ('python',), _python),
    RegisteredAdapter('typescript-compiler', ('javascript','typescript'), _typescript),
    RegisteredAdapter('powershell-ast', ('powershell',), _powershell),
    RegisteredAdapter('kotlin-compiler-psi', ('kotlin',), _kotlin),
    RegisteredAdapter('javac-compiler-ast', ('java',), _java),
    RegisteredAdapter('clang-compiler-ast', ('c','cpp','header'), _clang),
    RegisteredAdapter('go-stdlib-parser', ('go',), _go),
    RegisteredAdapter('ruby-ripper-parser', ('ruby',), _ruby),
    RegisteredAdapter('sql-structural-subset', ('sql',), _sql),
    RegisteredAdapter('xml-expat-source-structure', ('xml',), _xml),
    RegisteredAdapter('clojure-structural-forms', ('clojure',), _clojure),
    RegisteredAdapter('php-offline-token-parser', ('php',), _php),
    RegisteredAdapter('swift-compiler-ast', ('swift',), _swift),
    RegisteredAdapter('clang-objective-c-ast', ('objective_c',), _objc),
    RegisteredAdapter('shell-offline-source-structure', ('shell',), _offline_reader('shell')),
    RegisteredAdapter('rust-offline-source-structure', ('rust',), _offline_reader('rust')),
    RegisteredAdapter('csharp-offline-source-structure', ('csharp',), _offline_reader('csharp')),
    RegisteredAdapter('dart-offline-source-structure', ('dart',), _offline_reader('dart')),
    RegisteredAdapter('scala-offline-source-structure', ('scala',), _offline_reader('scala')),
    RegisteredAdapter('groovy-offline-source-structure', ('groovy',), _offline_reader('groovy')),
    RegisteredAdapter('lua-offline-source-structure', ('lua',), _offline_reader('lua')),
    RegisteredAdapter('r-offline-source-structure', ('r',), _offline_reader('r')),
    RegisteredAdapter('julia-offline-source-structure', ('julia',), _offline_reader('julia')),
    RegisteredAdapter('perl-offline-source-structure', ('perl',), _offline_reader('perl')),
    RegisteredAdapter('haskell-offline-source-structure', ('haskell',), _offline_reader('haskell')),
    RegisteredAdapter('erlang-offline-source-structure', ('erlang',), _offline_reader('erlang')),
    RegisteredAdapter('elixir-offline-source-structure', ('elixir',), _offline_reader('elixir')),
    RegisteredAdapter('fsharp-offline-source-structure', ('fsharp',), _offline_reader('fsharp')),
    RegisteredAdapter('visual_basic-offline-source-structure', ('visual_basic',), _offline_reader('visual_basic')),
    RegisteredAdapter('zig-offline-source-structure', ('zig',), _offline_reader('zig')),
    RegisteredAdapter('nim-offline-source-structure', ('nim',), _offline_reader('nim')),

)


def _safe_source_candidates(context: AdapterContext) -> tuple[dict[str, dict], list[dict]]:
    """Preflight every language path before invoking any native/compiler adapter.

    Inventory also rejects links, but direct caller/connector input must not
    bypass that policy. Reject symlinks at any component (including targets
    *inside* the source root). This is a safety preflight, not an OS-level
    atomic/openat sandbox and does not claim protection from a concurrent
    hostile mutation of the filesystem.
    """
    safe={};errors=[]
    root=context.root.resolve()
    for source,item in context.candidates.items():
        reason=None
        if not isinstance(source,str) or not source or ('\\' in source) or (':' in source) or '\x00' in source:
            reason='unsafe_source_path'
        else:
            parts=source.split('/')
            if (source.startswith('/') or any(x in ('','.','..') for x in parts) or
                    PurePosixPath(source).is_absolute()):
                reason='unsafe_source_path'
            else:
                lexical=root.joinpath(*parts)
                current=lexical
                try:
                    while current!=root:
                        if current.is_symlink():
                            reason='unsafe_source_path';break
                        current=current.parent
                    if reason is None:
                        target=lexical.resolve()
                        if not target.is_relative_to(root) or not target.is_file():
                            reason='unsafe_source_path'
                        elif target.stat().st_size>2_000_000:
                            reason='source_exceeds_inventory_limit'
                except (OSError,ValueError,RuntimeError):
                    reason='unsafe_source_path'
        if reason:
            errors.append({'path':source,'reason':reason,
                           'details':'source not accepted for adapter parsing',
                           'semantic_nodes_published':False})
        else:
            safe[source]=item
    return safe,errors


def run_adapters(context: AdapterContext) -> tuple[dict, list[dict], list[dict]]:
    allowed,preflight_warnings=_safe_source_candidates(context)
    safe_context=AdapterContext(context.root,allowed,context.graph_data)
    detected = {v.get('language','unknown') for v in allowed.values()}
    results = {}; warnings = list(preflight_warnings); report = []
    for adapter in REGISTRY:
        requested=sorted(detected.intersection(adapter.languages))
        if not requested:continue
        try:
            files, issues = adapter.analyze(safe_context)
        except Exception as exc:
            files = {}
            issues = [{'reason':'adapter_failed','adapter':adapter.name,
                       'path':p,'details':f'{type(exc).__name__}: {str(exc)[:220]}',
                       'semantic_nodes_published':False}
                      for p,v in allowed.items() if v.get('language') in adapter.languages]
        for p, graph in files.items():
            if p in results:
                warnings.append({'reason':'duplicate_semantic_adapter', 'path':p,'adapter':adapter.name})
                continue
            if p not in allowed or allowed[p].get('language') not in adapter.languages:
                warnings.append({'reason':'adapter_out_of_scope','path':p,'adapter':adapter.name})
                continue
            results[p] = graph
        warnings.extend(issues)
        report.append({'adapter':adapter.name,'languages':requested,
                       'parsed_files':sum(p in results for p in files),
                       'issues':len(issues)})
    # Source-only contract: no compilation or execution of analyzed projects.
    # The legacy Kotlin/JVM bytecode enrichment used kotlinc on user sources
    # and MUST NOT be called from static analysis. PSI and javac AST parse the
    # input as source without generating project bytecode.
    from .jni_symbols import connect_native_jni
    warnings.extend(connect_native_jni(results))
    from .jni_registration import connect_register_natives
    warnings.extend(connect_register_natives(context.root, allowed, results))
    return results, warnings, report
