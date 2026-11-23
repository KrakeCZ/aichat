"""Evidence-backed, literal local file references in additional ecosystems.

This deliberately does not interpret variables or execute code. Dynamic loads
cannot be resolved reliably without a runtime and remain out of scope.
"""
from __future__ import annotations
from dataclasses import dataclass
import posixpath
import re
from . import Context
from analyzer.model import ConfidenceLevel, EdgeType

# Patterns capture a literal path/module identifier, not arbitrary expressions.
PATTERNS = {
    "ruby": [re.compile(r"\brequire_relative\s*(?:\(\s*)?['\"]([^'\"]+)['\"]")],
    "php": [re.compile(r"\b(?:include|include_once|require|require_once)\s*(?:\(\s*)?['\"]([^'\"]+)['\"]")],
    "dart": [re.compile(r"\b(?:import|export|part)\s+['\"]([^'\"]+)['\"]")],
    "lua": [re.compile(r"\brequire\s*(?:\(\s*)?['\"]([^'\"]+)['\"]")],
    "r": [re.compile(r"\bsource\s*\(\s*['\"]([^'\"]+)['\"]")],
    "matlab": [re.compile(r"\b(?:run|addpath)\s*\(\s*['\"]([^'\"]+)['\"]")],
    "perl": [re.compile(r"\brequire\s*['\"]([^'\"]+)['\"]")],
}
EXTS = {"ruby":(".rb",), "php":(".php",), "dart":(".dart",),
        "lua":(".lua", "/init.lua"), "r":(".r", ".R"),
        "matlab":(".m",), "perl":(".pl", ".pm")}


def local_target(source: str, raw: str, lang: str, paths: set[str]) -> str | None:
    # URI imports and variable expansion are not accepted as static file references.
    if raw.startswith(("package:", "dart:", "/", "\\")) or "${" in raw or "$" in raw:
        return None
    name = raw
    if lang == "lua" and "/" not in raw and "\\" not in raw:
        name = raw.replace(".", "/")
    name = name.replace("\\", "/")
    base = posixpath.dirname(source)
    candidates = [posixpath.normpath(posixpath.join(base, name))]
    if lang == "lua":
        # Bare Lua `require 'pkg.mod'` can resolve from project's package root.
        candidates.append(posixpath.normpath(name))
    for candidate in candidates:
        if candidate == ".." or candidate.startswith("../"):
            continue
        possibles = [candidate]
        if not posixpath.splitext(candidate)[1]:
            possibles.extend(candidate+ext for ext in EXTS[lang] if not ext.startswith("/"))
            if lang == "lua":
                possibles.append(candidate + "/init.lua")
        for p in possibles:
            if p in paths and p != source:
                return p
    return None


@dataclass
class LocalReferencesPlugin:
    name: str = "local_references"
    languages: tuple[str, ...] = tuple(PATTERNS)

    def analyze(self, ctx: Context) -> None:
        for lang in self.languages:
            for source in ctx.files_by_language.get(lang, []):
                try:
                    content = ctx.read(source)
                except OSError:
                    continue
                lines = content.splitlines()
                for pattern in PATTERNS[lang]:
                    for match in pattern.finditer(content):
                        reference = match.group(1)
                        line = content.count("\n", 0, match.start()) + 1
                        target = local_target(source, reference, lang, ctx.paths)
                        if target:
                            ctx.add(source, target, line, lines[line-1], self.name,
                                    EdgeType.SCRIPT_SOURCE, ConfidenceLevel.RESOLVED_STATIC)
                        elif reference.startswith((".", "/")) and ":" not in reference:
                            ctx.miss(self.name, source, line, reference)


PLUGIN = LocalReferencesPlugin()
