"""Static Gradle module declarations for Atlas' architecture graph.

This reader never evaluates Gradle scripts. It recognizes only literal module
names, project dependencies and Android namespaces. Dynamic expressions are
left unresolved. Both Gradle Kotlin DSL and Groovy DSL are accepted.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .build_context import _lines, _not_inside_string, _cmake_declaration_context
from .android_variants import _blocks, _ancestors

_MODULE_LITERAL = re.compile(r'''["'](:[A-Za-z_][\w.:-]*)["']''')
_INCLUDE = re.compile(r'\binclude\s*(?:\(([^)]*)\)|\s+([^\n]+))')
_PROJECT_DEP = re.compile(
    r'\b(?:api|implementation|compileOnly|runtimeOnly|testImplementation|'
    r'androidTestImplementation|debugImplementation|releaseImplementation|ksp|kapt)'
    r'\s*(?:\(\s*)?project\s*\(\s*["\'](:[\w.:-]+)["\']\s*\)')
_NAMESPACE = re.compile(r'''\bnamespace\s*(?:=\s*|\s+)["']([A-Za-z_][\w.]*)["']''')
_CMAKE = re.compile(
    r'''\bpath\s*(?:=\s*)?(?:file\s*\(\s*)?["']([^"']+)["']\s*\)?''')


@dataclass
class GradleModuleInfo:
    name: str
    build_file: str
    namespace: str = ""
    dependencies: List[Tuple[str, int, str]] = field(default_factory=list)
    cmake_file: Optional[str] = None
    cmake_line: int = 0
    cmake_text: str = ""


class GradleAnalyzer:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir).resolve()
        self.modules: Dict[str, GradleModuleInfo] = {}
        self.settings_includes: List[Tuple[str, str, int, str]] = []

    def parse_settings(self, settings_rel_path: str, content: str) -> List[str]:
        modules = []
        for idx, code, original in _lines(content):
            for match in _INCLUDE.finditer(code):
                if not _not_inside_string(code, match.start()):
                    continue
                body = match.group(1) if match.group(1) is not None else match.group(2)
                for m in _MODULE_LITERAL.finditer(body):
                    name = m.group(1)
                    if name not in modules:
                        modules.append(name)
                        self.settings_includes.append((settings_rel_path, name, idx, original))
        return modules

    def parse_build_script(self, build_rel_path: str, content: str) -> GradleModuleInfo:
        parent = posixpath.dirname(build_rel_path.replace('\\', '/'))
        name = ':' + parent.replace('/', ':') if parent else ':root'
        lines = list(_lines(content))
        namespace_literals = []
        # Recognize namespace only inside a literal android { ... } block.
        # A comment or helper text outside that DSL is not an AGP namespace.
        clean = '\n'.join(code for _, code, _ in lines)
        blocks = _blocks(clean)
        line_offsets = []
        position = 0
        for _, code, _ in lines:
            line_offsets.append(position)
            position += len(code) + 1
        dependencies = []
        cmake_file = None
        cmake_line = 0
        cmake_text = ''
        for offset, (idx, code, original) in enumerate(lines):
            for m in _NAMESPACE.finditer(code):
                if not _not_inside_string(code, m.start()):
                    continue
                absolute_pos = line_offsets[offset] + m.start()
                if any(block.start < absolute_pos < block.end and
                       'android' in _ancestors(blocks, i)
                       for i, block in enumerate(blocks)):
                    namespace_literals.append(m.group(1))
            for m in _PROJECT_DEP.finditer(code):
                if _not_inside_string(code, m.start()):
                    dependencies.append((m.group(1), idx, original))
            for m in _CMAKE.finditer(code):
                if not _not_inside_string(code, m.start()):
                    continue
                if not _cmake_declaration_context(lines, offset, m.start()):
                    continue
                rel = m.group(1).replace('\\', '/')
                if rel.startswith('/') or '..' in rel.split('/') or '$' in rel or ':' in rel:
                    continue
                cmake_file = posixpath.normpath(posixpath.join(parent, rel))
                cmake_line, cmake_text = idx, original
        # Multiple distinct declarations are ambiguous (e.g. profile-dependent
        # DSL): never choose a namespace arbitrarily.
        namespace = namespace_literals[0] if len(set(namespace_literals)) == 1 else ''
        info = GradleModuleInfo(name=name, build_file=build_rel_path,
                                namespace=namespace, dependencies=dependencies,
                                cmake_file=cmake_file, cmake_line=cmake_line,
                                cmake_text=cmake_text)
        self.modules[name] = info
        return info
