"""Script Dependency Analyzer for Universal Code Atlas.

Extracts intra-repository dependencies for Python, Shell, and PowerShell scripts.
Only links verifiable, static relationships without inventing false edges.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PY_IMPORT_REGEX = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
    re.MULTILINE,
)

SH_SOURCE_REGEX = re.compile(
    r'^\s*(?:source|\.)\s+["\']?([^"\'\s;&]+)["\']?',
    re.MULTILINE,
)

PS1_DOT_SOURCE_REGEX = re.compile(
    r'^\s*(?:\.\s+|&\s+)["\']?([.\\\w-]+\.ps1)["\']?',
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ScriptDependency:
    target_path: str
    line: int
    text: str


class ScriptAnalyzer:
    def __init__(self, known_files: Set[str]):
        self.known_files = known_files

    def parse_python(self, rel_path: str, text: str) -> List[ScriptDependency]:
        """Resolve verifiable local Python imports (absolute and relative).

        No external site-packages, wildcard guesses, or dynamic import assumptions.
        """
        parent = posixpath.dirname(rel_path.replace("\\", "/"))
        deps: List[ScriptDependency] = []
        seen: Set[Tuple[str, int]] = set()

        def add_candidates(base: str, line: int, snippet: str) -> bool:
            module_path = base.replace(".", "/") if not base.startswith("/") else base
            for candidate in (module_path + ".py", posixpath.join(module_path, "__init__.py")):
                candidate = posixpath.normpath(candidate)
                if candidate in self.known_files and candidate != rel_path:
                    key = (candidate, line)
                    if key not in seen:
                        seen.add(key)
                        deps.append(ScriptDependency(target_path=candidate, line=line, text=snippet))
                    return True
            return False

        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            imp = re.match(r"^import\s+([\w.]+)", stripped)
            frm = re.match(r"^from\s+([\w.]*)\s+import\s+([\w*,\s]+)", stripped)
            if imp:
                module = imp.group(1)
                # Python can import `pkg.mod` from any directory on sys.path.
                # We intentionally try the repository root and current package.
                for candidate in (module.replace(".", "/"), posixpath.join(parent, module.replace(".", "/"))):
                    if add_candidates(candidate, line_no, stripped):
                        break
            elif frm:
                module, imported = frm.group(1), frm.group(2)
                if module.startswith("."):
                    leading = len(module) - len(module.lstrip("."))
                    base = parent
                    for _ in range(leading - 1):
                        base = posixpath.dirname(base)
                    dotted = module.lstrip(".")
                    target_base = posixpath.join(base, dotted.replace(".", "/")) if dotted else base
                    if dotted and add_candidates(target_base, line_no, stripped):
                        continue
                    for name in re.findall(r"[a-zA-Z_][\w]*", imported):
                        if name != "as":
                            add_candidates(posixpath.join(target_base, name), line_no, stripped)
                else:
                    for candidate in (module.replace(".", "/"), posixpath.join(parent, module.replace(".", "/"))):
                        if add_candidates(candidate, line_no, stripped):
                            break
                        # `from package import module`: module is often a file.
                        for name in re.findall(r"[a-zA-Z_][\w]*", imported):
                            add_candidates(posixpath.join(candidate, name), line_no, stripped)
        return deps

    def parse_shell(self, rel_path: str, text: str) -> List[ScriptDependency]:
        parent_dir = posixpath.dirname(rel_path.replace("\\", "/"))
        lines = text.splitlines()
        deps: List[ScriptDependency] = []

        for idx, line in enumerate(lines, 1):
            m = SH_SOURCE_REGEX.match(line)
            if m:
                target_name = m.group(1)
                candidate = posixpath.normpath(posixpath.join(parent_dir, target_name))
                if candidate in self.known_files:
                    deps.append(ScriptDependency(target_path=candidate, line=idx, text=line.strip()))

        return deps

    def parse_powershell(self, rel_path: str, text: str) -> List[ScriptDependency]:
        parent_dir = posixpath.dirname(rel_path.replace("\\", "/"))
        lines = text.splitlines()
        deps: List[ScriptDependency] = []

        for idx, line in enumerate(lines, 1):
            m = PS1_DOT_SOURCE_REGEX.match(line)
            if m:
                target_name = m.group(1).replace("\\", "/")
                candidate = posixpath.normpath(posixpath.join(parent_dir, target_name))
                if candidate in self.known_files:
                    deps.append(ScriptDependency(target_path=candidate, line=idx, text=line.strip()))

        return deps

