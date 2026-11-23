"""Go local imports resolved from go.mod module paths, without external network."""
from __future__ import annotations
import posixpath
import re
from dataclasses import dataclass
from . import Context
from analyzer.model import EdgeType, ConfidenceLevel

MODULE = re.compile(r'^\s*module\s+([^\s]+)', re.M)
IMPORT = re.compile(r'\bimport\s*(?:\(\s*([\s\S]*?)\)|(?:[\w.]+\s+)?"([^"]+)")')
QUOTES = re.compile(r'"([^"]+)"')


@dataclass
class GoPlugin:
    name: str = "go"
    languages: tuple[str, ...] = ("go",)

    def analyze(self, ctx: Context) -> None:
        mods = {}
        for modfile in [p for p in ctx.paths if p.endswith("go.mod")]:
            module = MODULE.search(ctx.read(modfile))
            if module:
                mods[module.group(1)] = posixpath.dirname(modfile)
        packages = {}
        for path in ctx.files_by_language.get("go", []):
            packages.setdefault(posixpath.dirname(path), []).append(path)
        for source in ctx.files_by_language.get("go", []):
            content = ctx.read(source)
            for match in IMPORT.finditer(content):
                quoted = [match.group(2)] if match.group(2) else QUOTES.findall(match.group(1) or "")
                line = content.count("\n", 0, match.start()) + 1
                for reference in quoted:
                    for module, folder in mods.items():
                        if reference.startswith(module + "/"):
                            dest = posixpath.normpath(posixpath.join(folder, reference[len(module)+1:]))
                            for target in packages.get(dest, []):
                                if target != source:
                                    ctx.add(source, target, line, content.splitlines()[line-1], self.name, EdgeType.IMPORT,
                                            ConfidenceLevel.RESOLVED_STATIC if len(packages.get(dest, [])) == 1 else ConfidenceLevel.CANDIDATE)
                            break


PLUGIN = GoPlugin()
