"""Shell scripts: local source / dot includes."""
from __future__ import annotations
import posixpath
import re
from dataclasses import dataclass
from . import Context
from analyzer.model import EdgeType
SOURCE = re.compile(r'^\s*(?:source|\.)\s+["\']?([^"\'\s;&]+)', re.M)


@dataclass
class ShellPlugin:
    name: str = "shell"
    languages: tuple[str, ...] = ("shell",)

    def analyze(self, ctx: Context) -> None:
        for source in ctx.files_by_language.get("shell", []):
            lines = ctx.read(source).splitlines()
            for no, line in enumerate(lines, 1):
                match = SOURCE.match(line)
                if not match:
                    continue
                value = match.group(1)
                if "$" in value:
                    continue
                path = posixpath.normpath(posixpath.join(posixpath.dirname(source), value))
                if path in ctx.paths and path != source:
                    ctx.add(source, path, no, line, self.name, EdgeType.SCRIPT_SOURCE)


PLUGIN = ShellPlugin()
