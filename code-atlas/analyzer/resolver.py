"""Master Architecture Resolver for Universal Code Atlas.

Coordinates file inventory, AST extraction, symbol table generation, and
cross-language resolution passes (Kotlin/Java, C++, JNI, Manifest, Resources,
Gradle, CMake, Scripts).
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .android_resources import AndroidResourceRegistry, R_RESOURCE_REGEX, VIEW_BINDING_REGEX
from .build_context import _lines, _mask_quoted
from .android_xml import AndroidXmlAnalyzer
from .cmake import CMakeAnalyzer
from .gradle import GradleAnalyzer
from .inventory import FileEntry, Inventory
from .jni import JniLinker, demangle_jni_name
from .kotlin_java import KotlinJavaAnalyzer
from .model import (
    ConfidenceLevel,
    Edge,
    EdgeEvidence,
    EdgeType,
    Graph,
    Node,
    NodeType,
)
from .native_cpp import NativeCppAnalyzer
from .scripts import ScriptAnalyzer
from .symbols import Symbol, SymbolTable


class ArchitectureResolver:
    def __init__(self, root_dir: Path, prefix: str = "", repository: str = ""):
        self.root_dir = Path(root_dir).resolve()
        self.prefix = prefix

        self.inventory = Inventory(self.root_dir, self.prefix)
        self.symbol_table = SymbolTable()
        self.resource_registry = AndroidResourceRegistry()

        self.kotlin_java_analyzer = KotlinJavaAnalyzer()
        self.cpp_analyzer = NativeCppAnalyzer()
        self.xml_analyzer = AndroidXmlAnalyzer()
        self.gradle_analyzer = GradleAnalyzer(self.root_dir)
        self.cmake_analyzer = CMakeAnalyzer(self.root_dir)
        self.jni_linker = JniLinker()

        self.graph = Graph(repository=repository)

        # Cached analyses
        self.kotlin_analyses: Dict[str, Any] = {}
        self.cpp_analyses: Dict[str, Any] = {}
        self.xml_analyses: Dict[str, Any] = {}
        self.gradle_analyses: Dict[str, Any] = {}
        self.cmake_analyses: Dict[str, Any] = {}

    def build_graph(self) -> Graph:
        return self.resolve()

    def resolve(self) -> Graph:
        # Phase 1: Scan inventory and create file nodes
        files = self.inventory.scan()
        self.graph.source_commit = self.inventory.source_commit
        # Visible exclusion ledger: Atlas exports are analysis products, not
        # source files to feed back into the main dependency graph.
        if self.inventory.excluded_atlas_exports:
            self.graph.metadata['excluded_atlas_export_roots'] = sorted(
                set(self.inventory.excluded_atlas_exports))
        if self.inventory.omitted_large_source_count:
            self.graph.metadata['omitted_large_source_count'] = self.inventory.omitted_large_source_count
            self.graph.metadata['omitted_large_sources'] = self.inventory.omitted_large_sources

        known_paths = {f.path for f in files}

        for f in files:
            node = Node(
                id=f.path,
                type=NodeType.FILE,
                name=f.name,
                category=f.category,
                path=f.path,
                properties={
                    "language": f.language,
                    "module": f.module,
                    "module_basis": "heuristic_file_layout",
                    "size_bytes": f.size_bytes,
                    "is_test": f.is_test,
                },
            )
            self.graph.add_node(node)

            # Resource ownership is registered after Gradle module mapping.

        # Parse literal Gradle module configuration *before* reading XML. AGP 8+
        # AndroidManifest.xml files often omit package and rely on the module's
        # declared namespace. No Gradle scripts are executed.
        for f in files:
            if f.name in ('settings.gradle', 'settings.gradle.kts'):
                self.gradle_analyzer.parse_settings(f.path, (self.root_dir / f.path).read_text('utf-8', errors='replace'))
        for f in files:
            if f.name in ('build.gradle', 'build.gradle.kts'):
                info = self.gradle_analyzer.parse_build_script(
                    f.path, (self.root_dir / f.path).read_text('utf-8', errors='replace'))
                self.gradle_analyses[f.path] = info

        # The longest matching build-script parent is a declared source-tree
        # ownership candidate. It is not an evaluated Gradle variant model.
        module_roots = sorted(((posixpath.dirname(info.build_file), name, info)
                               for name, info in self.gradle_analyzer.modules.items()),
                              key=lambda x: len(x[0]), reverse=True)
        for f in files:
            for directory, name, info in module_roots:
                if (directory and f.path.startswith(directory + '/')) or (
                    not directory and not any(
                        d and f.path.startswith(d + '/') for d, _, _ in module_roots)):
                    f.module = name
                    self.graph.nodes[f.path].properties.update(
                        {'module': name, 'module_basis': 'static_gradle_build_directory'})
                    break
        self.xml_analyzer.default_namespaces = {
            name: info.namespace for name, info in self.gradle_analyzer.modules.items()
            if info.namespace}
        for f in files:
            if f.is_resource:
                self.resource_registry.register_resource_file(f.path, f.module)

        # Phase 2: Parse individual files and populate symbol table
        external_methods: Dict[str, Tuple[str, int, str]] = {}  # fqn -> (path, line, snippet)

        for f in files:
            abs_p = self.root_dir / f.path
            source_bytes = abs_p.read_bytes()

            if f.language in ("kotlin", "java"):
                ana = self.kotlin_java_analyzer.parse_file(f.path, source_bytes)
                self.kotlin_analyses[f.path] = ana

                # Update node metadata
                node = self.graph.nodes[f.path]
                node.package = ana.package
                node.symbols = [s.to_dict() for s in ana.symbols]
                node.search_symbols = ana.search_symbols

                # Register in symbol table
                kt_lines = source_bytes.decode("utf-8", errors="replace").splitlines()
                for sym in ana.symbols:
                    self.symbol_table.register(sym)
                    if sym.is_external:
                        line_snippet = kt_lines[sym.line - 1].strip() if 1 <= sym.line <= len(kt_lines) else sym.name
                        external_methods[sym.fqn] = (f.path, sym.line, line_snippet)

            elif f.language in ("cpp", "c", "header"):
                ana = self.cpp_analyzer.parse_file(f.path, source_bytes)
                self.cpp_analyses[f.path] = ana

                node = self.graph.nodes[f.path]
                node.symbols = [s.to_dict() for s in ana.symbols]
                node.search_symbols = ana.search_symbols

                for sym in ana.symbols:
                    self.symbol_table.register(sym)

            elif f.language == "xml":
                ana = self.xml_analyzer.parse_file(f.path, source_bytes, f.module)
                self.xml_analyses[f.path] = ana

                node = self.graph.nodes[f.path]
                node.package = ana.package
                node.search_symbols = ana.search_symbols

            # Gradle scripts were processed in the pre-pass above. In
            # particular, ordinary .gradle files are not silently ignored.

            elif f.language == "cmake":
                text = source_bytes.decode("utf-8", errors="replace")
                c_ana = self.cmake_analyzer.parse_cmake_file(f.path, text)
                self.cmake_analyses[f.path] = c_ana

        # Phase 3: Resolution Passes

        # Pass 3A: Kotlin & Java Import Resolution & Wildcard Handling
        for file_path, ana in self.kotlin_analyses.items():
            wildcard_packages: Set[str] = set()
            known_imports: Dict[str, str] = {}  # alias or simple_name -> target file_path

            for imp in ana.imports:
                if imp.is_wildcard:
                    pkg = imp.name.rsplit(".*", 1)[0]
                    wildcard_packages.add(pkg)
                    continue

                # Exact or aliased import
                targets = self.symbol_table.lookup_fqn(imp.name)
                if targets:
                    target_file = targets[0].file_path
                    self.graph.add_edge(
                        source=file_path,
                        target=target_file,
                        edge_type=EdgeType.IMPORT_ALIAS if imp.alias else EdgeType.IMPORT,
                        confidence=ConfidenceLevel.RESOLVED_STATIC,
                        analyzer="kotlin_java",
                        evidence=EdgeEvidence(path=file_path, line=imp.line, snippet=imp.text),
                    )
                    symbol_key = imp.alias or imp.name.rsplit(".", 1)[-1]
                    known_imports[symbol_key] = target_file
                else:
                    # Check if imported name matches a class whose FQN is in project
                    by_name = self.symbol_table.lookup_by_name(imp.name.rsplit(".", 1)[-1])
                    if by_name:
                        matched = [s for s in by_name if s.fqn == imp.name]
                        if matched:
                            target_file = matched[0].file_path
                            self.graph.add_edge(
                                source=file_path,
                                target=target_file,
                                edge_type=EdgeType.IMPORT,
                                confidence=ConfidenceLevel.RESOLVED_STATIC,
                                analyzer="kotlin_java",
                                evidence=EdgeEvidence(path=file_path, line=imp.line, snippet=imp.text),
                            )
                            known_imports[imp.name.rsplit(".", 1)[-1]] = target_file

            # Pass 3B: Same-Package & Wildcard-Matched Type References
            for ref in ana.type_references:
                # 1. Check if type is an explicit qualified reference: a.b.c.Foo
                if "." in ref.name and ref.context == "qualified":
                    fqn_matches = self.symbol_table.lookup_fqn(ref.name)
                    if fqn_matches:
                        target_file = fqn_matches[0].file_path
                        self.graph.add_edge(
                            source=file_path,
                            target=target_file,
                            edge_type=EdgeType.TYPE_REFERENCE,
                            confidence=ConfidenceLevel.EXACT_SYNTAX,
                            analyzer="kotlin_java",
                            evidence=EdgeEvidence(path=file_path, line=ref.line, column=ref.column, snippet=ref.text),
                        )
                        continue

                # 2. Check known imports
                simple_name = ref.name.rsplit(".", 1)[-1]
                if simple_name in known_imports:
                    target_file = known_imports[simple_name]
                    edge_kind = EdgeType.TYPE_REFERENCE
                    if ref.context == "inheritance":
                        edge_kind = EdgeType.INHERITANCE
                    elif ref.context == "class_literal":
                        edge_kind = EdgeType.TYPE_REFERENCE
                    self.graph.add_edge(
                        source=file_path,
                        target=target_file,
                        edge_type=edge_kind,
                        confidence=ConfidenceLevel.RESOLVED_STATIC,
                        analyzer="kotlin_java",
                        evidence=EdgeEvidence(path=file_path, line=ref.line, column=ref.column, snippet=ref.text),
                    )
                    continue

                # 3. Check same package
                same_pkg_matches = self.symbol_table.lookup_same_package(ana.package, simple_name)
                if same_pkg_matches:
                    # Filter out declarations from the current file itself
                    external_candidates = [m for m in same_pkg_matches if m.file_path != file_path]
                    candidate_files = {m.file_path for m in external_candidates}
                    if len(candidate_files) == 1:
                        target_file = next(iter(candidate_files))
                        edge_kind = EdgeType.SAME_PACKAGE
                        if ref.context == "inheritance":
                            edge_kind = EdgeType.INHERITANCE
                        self.graph.add_edge(
                            source=file_path,
                            target=target_file,
                            edge_type=edge_kind,
                            confidence=ConfidenceLevel.RESOLVED_STATIC,
                            analyzer="kotlin_java",
                            evidence=EdgeEvidence(path=file_path, line=ref.line, column=ref.column, snippet=ref.text),
                        )
                        continue
                    elif len(candidate_files) == 0:
                        # Local declaration within the same file (e.g., companion object, inner class)
                        continue
                    else:
                        # Ambiguous symbol across multiple files in the same package (e.g., Companion, Error)
                        # Do not blindly select matches[0] - preserve strict precision
                        continue

                # 4. Check wildcard imports
                wc_candidates: List[Any] = []
                for wc_pkg in wildcard_packages:
                    wc_matches = self.symbol_table.lookup_same_package(wc_pkg, simple_name)
                    wc_candidates.extend(m for m in wc_matches if m.file_path != file_path)
                wc_files = {m.file_path for m in wc_candidates}
                if len(wc_files) == 1:
                    target_file = next(iter(wc_files))
                    self.graph.add_edge(
                        source=file_path,
                        target=target_file,
                        edge_type=EdgeType.IMPORT_WILDCARD,
                        confidence=ConfidenceLevel.RESOLVED_STATIC,
                        analyzer="kotlin_java",
                        evidence=EdgeEvidence(path=file_path, line=ref.line, column=ref.column, snippet=ref.text),
                    )
                    continue

            # Pass 3C: Resource & ViewBinding references in Kotlin/Java
            abs_kt = self.root_dir / file_path
            kt_text = abs_kt.read_text(encoding="utf-8", errors="replace")
            for idx, line, original in _lines(kt_text):
                # Strip quoted text and comments; a resource name mentioned in
                # a log message or a comment is not an actual reference.
                code = _mask_quoted(line)
                module = self.graph.nodes[file_path].properties['module']
                for rm in R_RESOURCE_REGEX.finditer(code):
                    rtype, rname = rm.group(1), rm.group(2)
                    target_res_files = self.resource_registry.find_resource(rtype, rname, module)
                    for tr in target_res_files:
                        self.graph.add_edge(
                            source=file_path,
                            target=tr,
                            edge_type=EdgeType.RESOURCE_REFERENCE,
                            confidence=(ConfidenceLevel.RESOLVED_STATIC if len(target_res_files) == 1
                                        else ConfidenceLevel.CANDIDATE),
                            analyzer="android_resources",
                            evidence=EdgeEvidence(path=file_path, line=idx, snippet=original),
                        )

                # ViewBinding: link only if exactly one layout is in the source
                # module; same-name layouts from other modules are unrelated.
                for vm in VIEW_BINDING_REGEX.finditer(code):
                    b_class = vm.group(1)
                    target_layout = self.resource_registry.find_layout_by_binding(b_class, module)
                    if target_layout:
                        self.graph.add_edge(
                            source=file_path,
                            target=target_layout,
                            edge_type=EdgeType.VIEW_BINDING,
                            confidence=ConfidenceLevel.RESOLVED_STATIC,
                            analyzer="android_resources",
                            evidence=EdgeEvidence(path=file_path, line=idx, snippet=original),
                        )

                # System.loadLibrary("...")
                for lm in self.jni_linker.extract_load_library_calls(line):
                    lib_name, line_no, snip = lm
                    # Match against CMake targets or output names
                    for c_path, c_ana in self.cmake_analyses.items():
                        if lib_name in c_ana.targets:
                            self.graph.add_edge(
                                source=file_path,
                                target=c_path,
                                edge_type=EdgeType.NATIVE_LIBRARY_LOAD,
                                confidence=ConfidenceLevel.RESOLVED_STATIC,
                                analyzer="jni",
                                evidence=EdgeEvidence(path=file_path, line=idx, snippet=snip),
                            )

        # Pass 3D: AndroidManifest.xml & Layout XML Linking
        for file_path, x_ana in self.xml_analyses.items():
            if x_ana.is_manifest:
                for comp in x_ana.manifest_components:
                    matches = self.symbol_table.lookup_fqn(comp.resolved_fqn)
                    # A simple-name match is not evidence of class identity:
                    # large Android projects often contain many MainActivity
                    # classes in unrelated modules. Do not manufacture edges.
                    if len({match.file_path for match in matches}) == 1:
                        target_file = matches[0].file_path
                        self.graph.add_edge(
                            source=file_path,
                            target=target_file,
                            edge_type=EdgeType.MANIFEST_COMPONENT,
                            confidence=ConfidenceLevel.RESOLVED_STATIC,
                            analyzer="android_xml",
                            evidence=EdgeEvidence(path=file_path, line=comp.line, snippet=comp.text),
                        )
            else:
                # Custom Views in layout
                for cv in x_ana.custom_views:
                    matches = self.symbol_table.lookup_fqn(cv.class_fqn)
                    if not matches:
                        simple_name = cv.class_fqn.rsplit(".", 1)[-1]
                        matches = self.symbol_table.lookup_by_name(simple_name)
                    if matches:
                        target_file = matches[0].file_path
                        self.graph.add_edge(
                            source=file_path,
                            target=target_file,
                            edge_type=EdgeType.TYPE_REFERENCE,
                            confidence=ConfidenceLevel.RESOLVED_STATIC,
                            analyzer="android_xml",
                            evidence=EdgeEvidence(path=file_path, line=cv.line, snippet=cv.text),
                        )

                # Layout includes
                for inc in x_ana.layout_includes:
                    target_lays = self.resource_registry.find_resource(
                        "layout", inc.target_layout, self.graph.nodes[file_path].properties['module'])
                    for tl in target_lays:
                        self.graph.add_edge(
                            source=file_path,
                            target=tl,
                            edge_type=EdgeType.RESOURCE_REFERENCE,
                            confidence=(ConfidenceLevel.RESOLVED_STATIC if len(target_lays) == 1
                                        else ConfidenceLevel.CANDIDATE),
                            analyzer="android_xml",
                            evidence=EdgeEvidence(path=file_path, line=inc.line, snippet=inc.text),
                        )

            # XML Resource references (@drawable/..., @xml/..., @menu/..., @mipmap/..., @layout/...)
            for r_ref in x_ana.resource_refs:
                target_res_files = self.resource_registry.find_resource(
                    r_ref.res_type, r_ref.res_name, self.graph.nodes[file_path].properties['module'])
                for tr in target_res_files:
                    if tr != file_path:
                        self.graph.add_edge(
                            source=file_path,
                            target=tr,
                            edge_type=EdgeType.RESOURCE_REFERENCE,
                            confidence=(ConfidenceLevel.RESOLVED_STATIC if len(target_res_files) == 1
                                        else ConfidenceLevel.CANDIDATE),
                            analyzer="android_xml",
                            evidence=EdgeEvidence(path=file_path, line=r_ref.line, snippet=r_ref.text),
                        )

        # Pass 3E: JNI Bridge Linking (C++ JNIEXPORT <-> Kotlin external fun)
        # Guarantees 1:1 method pairing and dual-sided structured evidence on both ends
        for cpp_path, cpp_ana in self.cpp_analyses.items():
            for exp in cpp_ana.jni_exports:
                match = self.jni_linker.match_jni_export(exp.mangled_name, external_methods)
                if match:
                    kt_file, kt_line, kt_snip = match
                    kt_ev = EdgeEvidence(
                        path=kt_file,
                        line=kt_line,
                        snippet=f"[Kotlin external fun] {kt_snip}",
                    )
                    cpp_ev = EdgeEvidence(
                        path=cpp_path,
                        line=exp.line,
                        snippet=f"[C++ JNIEXPORT] {exp.text}",
                    )
                    edge = self.graph.add_edge(
                        source=kt_file,
                        target=cpp_path,
                        edge_type=EdgeType.JNI_BINDING,
                        confidence=ConfidenceLevel.EXACT_SYNTAX,
                        analyzer="jni",
                        evidence=kt_ev,
                    )
                    if edge is not None:
                        # Append C++ side evidence if not present
                        if not any(e.path == cpp_ev.path and e.line == cpp_ev.line for e in edge.evidence):
                            edge.evidence.append(cpp_ev)
                        demangled = demangle_jni_name(exp.mangled_name)
                        method_name = demangled[2] if demangled else exp.mangled_name
                        if "methods" not in edge.properties:
                            edge.properties["methods"] = []
                        if not any(m.get("mangled_name") == exp.mangled_name for m in edge.properties["methods"]):
                            edge.properties["methods"].append({
                                "method_name": method_name,
                                "mangled_name": exp.mangled_name,
                                "kotlin_file": kt_file,
                                "kotlin_line": kt_line,
                                "kotlin_snippet": kt_snip,
                                "cpp_file": cpp_path,
                                "cpp_line": exp.line,
                                "cpp_snippet": exp.text,
                            })

        # Pass 3F: C/C++ Header Includes
        for cpp_path, cpp_ana in self.cpp_analyses.items():
            parent_dir = posixpath.dirname(cpp_path.replace("\\", "/"))
            for inc in cpp_ana.includes:
                inc_name = inc.name
                # Check relative to directory
                cand1 = posixpath.normpath(posixpath.join(parent_dir, inc_name))
                if cand1 in known_paths:
                    self.graph.add_edge(
                        source=cpp_path,
                        target=cand1,
                        edge_type=EdgeType.INCLUDE,
                        confidence=ConfidenceLevel.EXACT_SYNTAX,
                        analyzer="native_cpp",
                        evidence=EdgeEvidence(path=cpp_path, line=inc.line, snippet=inc.text),
                    )
                    continue

                # Check ends with inc_name
                candidates = [p for p in known_paths if p.endswith("/" + inc_name) or p == inc_name]
                if len(candidates) == 1:
                    self.graph.add_edge(
                        source=cpp_path,
                        target=candidates[0],
                        edge_type=EdgeType.INCLUDE,
                        confidence=ConfidenceLevel.RESOLVED_STATIC,
                        analyzer="native_cpp",
                        evidence=EdgeEvidence(path=cpp_path, line=inc.line, snippet=inc.text),
                    )

        # Pass 3G: CMake and Gradle Build System Links
        # settings.gradle.kts module includes (e.g. :module-a, :module-b)
        for s_path, target_mod, line_no, snip in self.gradle_analyzer.settings_includes:
            if target_mod in self.gradle_analyzer.modules:
                target_b = self.gradle_analyzer.modules[target_mod].build_file
                self.graph.add_edge(
                    source=s_path,
                    target=target_b,
                    edge_type=EdgeType.GRADLE_DEPENDENCY,
                    confidence=ConfidenceLevel.EXACT_SYNTAX,
                    analyzer="gradle",
                    evidence=EdgeEvidence(path=s_path, line=line_no, snippet=snip),
                )

        # Only declare Gradle edges whose references occur explicitly in
        # settings.gradle or a module dependency. File co-location does not
        # prove a dependency and must never create a fabricated evidence line.

        for g_path, g_info in self.gradle_analyses.items():
            # Inter-module dependencies: implementation(project(":lib"))
            for target_mod, line_no, snip in g_info.dependencies:
                if target_mod in self.gradle_analyzer.modules:
                    target_b = self.gradle_analyzer.modules[target_mod].build_file
                    self.graph.add_edge(
                        source=g_path,
                        target=target_b,
                        edge_type=EdgeType.GRADLE_DEPENDENCY,
                        confidence=ConfidenceLevel.EXACT_SYNTAX,
                        analyzer="gradle",
                        evidence=EdgeEvidence(path=g_path, line=line_no, snippet=snip),
                    )

            # Link to CMakeLists.txt
            if g_info.cmake_file and g_info.cmake_file in known_paths:
                self.graph.add_edge(
                    source=g_path,
                    target=g_info.cmake_file,
                    edge_type=EdgeType.CMAKE_SOURCE,
                    confidence=ConfidenceLevel.EXACT_SYNTAX,
                    analyzer="gradle",
                    evidence=EdgeEvidence(path=g_path, line=g_info.cmake_line, snippet=g_info.cmake_text),
                )

        # CMake target sources
        for c_path, c_ana in self.cmake_analyses.items():
            for target_name, target in c_ana.targets.items():
                for src_path, line_no, snip in target.sources:
                    if src_path in known_paths:
                        self.graph.add_edge(
                            source=c_path,
                            target=src_path,
                            edge_type=EdgeType.CMAKE_SOURCE,
                            confidence=ConfidenceLevel.EXACT_SYNTAX,
                            analyzer="cmake",
                            evidence=EdgeEvidence(path=c_path, line=line_no, snippet=snip),
                        )

        # Pass 3H: language plugins run separately in atlas.py (autodetected).
        # Legacy Android/Kotlin/C++/Gradle/CMake passes above remain untouched.

        return self.graph

    def get_parser_stats(self) -> Dict[str, Any]:
        return {
            "kotlin_java": self.kotlin_java_analyzer.get_stats(),
            "native_cpp": self.cpp_analyzer.get_stats(),
        }
