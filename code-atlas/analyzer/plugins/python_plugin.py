"""Python AST imports: absolute, relative, src layout, namespace packages."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import posixpath
from . import Context
from analyzer.model import EdgeType, ConfidenceLevel


def _module_file(base: str, known: set[str]) -> str | None:
    base = base.strip("/")
    for f in (base + ".py", base + "/__init__.py"):
        if f in known:
            return f
    return None


@dataclass
class PythonPlugin:
    name: str = "python"
    languages: tuple[str, ...] = ("python",)

    def analyze(self, ctx: Context) -> None:
        files = ctx.files_by_language.get("python", [])
        # Find plausible import roots. Supports src/, python/, backend/src/, etc.
        roots = {""}
        for f in files:
            parts = f.split("/")
            if "src" in parts[:-1]:
                roots.add("/".join(parts[:parts.index("src")+1]))
            if "python" in parts[:-1]:
                roots.add("/".join(parts[:parts.index("python")+1]))
        for source in files:
            try:
                content = ctx.read(source)
                tree = ast.parse(content, filename=source)
            except (OSError, SyntaxError, UnicodeError):
                continue
            parent = posixpath.dirname(source)
            lines = content.splitlines()

            def link(mod: str, line: int, relative_level: int = 0, specific: str = "") -> bool:
                if relative_level:
                    base = parent
                    for _ in range(relative_level - 1):
                        base = posixpath.dirname(base)
                    start = posixpath.join(base, mod.replace(".", "/")) if mod else base
                    bases = [start]
                else:
                    path = mod.replace(".", "/")
                    bases = [posixpath.join(root, path) for root in sorted(roots, key=len, reverse=True)]
                    bases.append(posixpath.join(parent, path))
                seen = set()
                for base in bases:
                    base = posixpath.normpath(base)
                    if base.startswith("../") or base in seen:
                        continue
                    seen.add(base)
                    # `from pkg import helper` can refer to pkg/helper.py.
                    if specific and specific != "*":
                        leaf = _module_file(base + "/" + specific, ctx.paths)
                        if leaf and leaf != source:
                            ctx.add(source, leaf, line, lines[line-1], self.name,
                                    EdgeType.IMPORT, ConfidenceLevel.RESOLVED_STATIC)
                            return True
                    target = _module_file(base, ctx.paths)
                    if target and target != source:
                        ctx.add(source, target, line, lines[line-1], self.name,
                                EdgeType.IMPORT, ConfidenceLevel.RESOLVED_STATIC)
                        return True
                return False

            for item in ast.walk(tree):
                if isinstance(item, ast.Import):
                    for alias in item.names:
                        link(alias.name, item.lineno)
                elif isinstance(item, ast.ImportFrom):
                    mod = item.module or ""
                    # Never invent a dependency to an external Python package.
                    linked = False
                    for alias in item.names:
                        if link(mod, item.lineno, item.level, alias.name):
                            linked = True
                    if not linked:
                        link(mod, item.lineno, item.level)
                        if item.level:
                            ctx.miss(self.name, source, item.lineno, "." * item.level + mod)


PLUGIN = PythonPlugin()
