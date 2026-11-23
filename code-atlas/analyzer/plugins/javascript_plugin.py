"""JS/TS ESM/CommonJS local imports, tsconfig alias and local workspaces."""
from __future__ import annotations
import json
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from . import Context
from .javascript_syntax import imports as scan_imports
from analyzer.model import EdgeType

# Supports multiline imports, side-effect imports, export-from, dynamic imports, require.
IMPORTS = re.compile(
    r"\b(?:import|export)\s+(?:type\s+)?(?:[\s\S]{0,500}?\s+from\s*)?[\"']([^\"']+)[\"']"
    r"|\b(?:import|require)\s*\(\s*[\"']([^\"']+)[\"']\s*\)",
    re.MULTILINE,
)
EXT = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")


def parse_jsonc(text: str) -> dict:
    # Remove JSONC comments without corrupting quoted URL strings.
    text = re.sub(r'("(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/)',
                  lambda m: m.group(1) if m.group(1).startswith('"') else "", text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return {}


def resolve_path(path: str, paths: set[str]) -> str | None:
    p = posixpath.normpath(path)
    if p.startswith("../") or p == "..":
        return None
    candidates = [p]
    stem, ext = posixpath.splitext(p)
    if ext in (".js", ".mjs", ".cjs"):
        candidates = [stem + e for e in EXT] + candidates
    if not ext:
        candidates.extend(p + e for e in EXT)
    candidates.extend(p.rstrip("/") + "/index" + e for e in EXT)
    return next((q for q in candidates if q in paths), None)


@dataclass
class JavaScriptPlugin:
    name: str = "javascript_typescript"
    languages: tuple[str, ...] = ("javascript", "typescript")

    def analyze(self, ctx: Context) -> None:
        configs = {}
        packages = {}
        for path in ctx.paths:
            if path.endswith(("tsconfig.json", "jsconfig.json")):
                configs[path] = parse_jsonc(ctx.read(path))
            if path.endswith("package.json"):
                try:
                    cfg = json.loads(ctx.read(path))
                    if cfg.get("name"):
                        packages[cfg["name"]] = posixpath.dirname(path)
                except (ValueError, OSError):
                    pass
        for source in ctx.files_by_language.get("javascript", []) + ctx.files_by_language.get("typescript", []):
            try:
                content = ctx.read(source)
            except OSError:
                continue
            directory = posixpath.dirname(source)
            config_path = next((p for p in sorted(configs, key=lambda s: len(posixpath.dirname(s)), reverse=True)
                                if not posixpath.dirname(p) or source.startswith(posixpath.dirname(p) + "/")), None)
            config = configs.get(config_path, {})
            cfgdir = posixpath.dirname(config_path) if config_path else ""
            compiler = config.get("compilerOptions", {})
            base = posixpath.normpath(posixpath.join(cfgdir, compiler.get("baseUrl", ".")))
            for ref, line in scan_imports(content):
                target = None
                if ref.startswith("."):
                    target = resolve_path(posixpath.join(directory, ref), ctx.paths)
                    if not target:
                        ctx.miss(self.name, source, line, ref)
                else:
                    for alias, values in compiler.get("paths", {}).items():
                        values = values if isinstance(values, list) else [values]
                        if "*" in alias:
                            prefix, suffix = alias.split("*", 1)
                            if not (ref.startswith(prefix) and ref.endswith(suffix)):
                                continue
                            variable = ref[len(prefix):len(ref)-len(suffix) if suffix else None]
                        elif ref == alias:
                            variable = ""
                        else:
                            continue
                        for value in values:
                            if isinstance(value, str):
                                target = resolve_path(posixpath.join(base, value.replace("*", variable)), ctx.paths)
                                if target:
                                    break
                        if target:
                            break
                    # baseUrl imports and local monorepo workspace packages
                    if not target and config_path:
                        target = resolve_path(posixpath.join(base, ref), ctx.paths)
                    if not target:
                        for package, folder in packages.items():
                            if ref == package or ref.startswith(package + "/"):
                                postfix = ref[len(package):].lstrip("/")
                                target = resolve_path(posixpath.join(folder, postfix or "src/index"), ctx.paths)
                                if target:
                                    break
                if target and target != source:
                    ctx.add(source, target, line, content.splitlines()[line-1], self.name, EdgeType.IMPORT)


PLUGIN = JavaScriptPlugin()
