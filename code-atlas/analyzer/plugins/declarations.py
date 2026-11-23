"""Conservative, local-only declaration extraction for common languages.

Indexing symbols is not the same as resolving calls. Python inheritance edges
are emitted only when an imported base class is found in an indexed module.
No project source is executed.
"""
from __future__ import annotations
import ast
from collections import Counter
import posixpath
import re
from . import Context
from analyzer.model import ConfidenceLevel, EdgeType

# Regex syntax recognizers for *declarations*, not complete language parsers.
PATTERNS = {
    "javascript": [
        ("class", r"\b(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"),
        ("function", r"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)"),
    ],
    "typescript": [
        ("class", r"\b(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"),
        ("interface", r"\b(?:export\s+)?interface\s+(\w+)"),
        ("type", r"\b(?:export\s+)?type\s+(\w+)\s*="),
        ("enum", r"\b(?:export\s+)?(?:const\s+)?enum\s+(\w+)"),
        ("function", r"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+(\w+)"),
    ],
    "rust": [
        ("struct", r"\b(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)"),
        ("enum", r"\b(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)"),
        ("trait", r"\b(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)"),
        ("function", r"\b(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"),
    ],
    "go": [
        ("struct", r"\btype\s+(\w+)\s+struct\s*\{"),
        ("interface", r"\btype\s+(\w+)\s+interface\s*\{"),
        ("function", r"\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\("),
    ],
    "powershell": [
        ("class", r"(?im)^\s*class\s+([\w-]+)\b"),
        ("function", r"(?im)^\s*function\s+([\w-]+)\b"),
        ("enum", r"(?im)^\s*enum\s+([\w-]+)\b"),
    ],
    "shell": [("function", r"(?m)^\s*(?:function\s+)?([\w-]+)\s*\(\s*\)\s*\{")],
    "csharp": [("class", r"\bclass\s+(\w+)"), ("interface", r"\binterface\s+(\w+)"),
               ("struct", r"\bstruct\s+(\w+)"), ("enum", r"\benum\s+(\w+)")],
    "swift": [("class", r"\bclass\s+(\w+)"), ("struct", r"\bstruct\s+(\w+)"),
              ("protocol", r"\bprotocol\s+(\w+)"), ("function", r"\bfunc\s+(\w+)")],
    "php": [("class", r"\bclass\s+(\w+)"), ("function", r"\bfunction\s+(\w+)")],
    "ruby": [("class", r"(?m)^\s*class\s+(\w+)"), ("module", r"(?m)^\s*module\s+(\w+)"),
             ("function", r"(?m)^\s*def\s+(\w+)")],
    "dart": [("class", r"\bclass\s+(\w+)"), ("function", r"\b(?:void|Future<[^>]+>)\s+(\w+)\s*\(")],
}


def _python_symbols(ctx: Context, source: str, tree: ast.AST) -> list[dict]:
    symbols: list[dict] = []
    def add(n: ast.AST, kind: str, name: str, *, top: bool = False) -> None:
        if len(symbols) >= 180:
            return
        symbols.append({"name": name, "kind": kind, "fqn": name,
                        "line": n.lineno, "is_top_level": top, "is_external": False})
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef):
            add(node, "class", node.name, top=True)
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(method, "method", f"{node.name}.{method.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, "function", node.name, top=True)
    return symbols


def _python_module(source: str, module: str, level: int, paths: set[str]) -> str | None:
    parent = posixpath.dirname(source)
    if level:
        base = parent
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        options = [posixpath.join(base, module.replace(".", "/"))]
    else:
        fragment = module.replace(".", "/")
        roots = [""]
        pieces = source.split("/")
        for component in ("src", "python"):
            if component in pieces[:-1]:
                roots.append("/".join(pieces[:pieces.index(component)+1]))
        roots.append(parent)
        options = [posixpath.join(root, fragment) for root in roots]
    for opt in options:
        opt = posixpath.normpath(opt)
        if opt == ".." or opt.startswith("../"):
            continue
        for candidate in (opt + ".py", opt.rstrip("/") + "/__init__.py"):
            if candidate in paths:
                return candidate
    return None


def _python_inheritance(ctx: Context, source: str, tree: ast.AST, class_names: dict[str, set[str]], lines: list[str]) -> int:
    imported: dict[str, str] = {}
    for item in getattr(tree, "body", []):
        if isinstance(item, ast.ImportFrom):
            mod = item.module or ""
            path = _python_module(source, mod, item.level, ctx.paths)
            for alias in item.names:
                # from pkg import submod.Base is uncommon but handled by module lookup
                target = _python_module(source, f"{mod}.{alias.name}".strip("."), item.level, ctx.paths)
                if target:
                    imported[alias.asname or alias.name] = target
                elif path:
                    imported[alias.asname or alias.name] = path
        elif isinstance(item, ast.Import):
            for alias in item.names:
                target = _python_module(source, alias.name, 0, ctx.paths)
                if target:
                    imported[alias.asname or alias.name.split(".")[0]] = target
    found = 0
    for item in getattr(tree, "body", []):
        if not isinstance(item, ast.ClassDef):
            continue
        for base in item.bases:
            if isinstance(base, ast.Name):
                ref = base.id
                target = imported.get(ref)
                expected = ref
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                ref = base.value.id
                target = imported.get(ref)
                expected = base.attr
            else:
                continue
            if target and target != source and expected in class_names.get(target, set()):
                before = len(ctx.graph.edges)
                ctx.add(source, target, item.lineno, lines[item.lineno-1], "python_classes",
                        EdgeType.INHERITANCE, ConfidenceLevel.RESOLVED_STATIC)
                if len(ctx.graph.edges) > before:
                    found += 1
    return found


def enrich_declarations(ctx: Context) -> dict:
    totals = Counter()
    python_cache: dict[str, tuple[ast.AST, list[str]]] = {}
    python_classes: dict[str, set[str]] = {}
    # Read only indexed first-party files, no execution of untrusted source.
    for lang, paths in ctx.files_by_language.items():
        for source in paths:
            try:
                content = ctx.read(source)
            except (OSError, UnicodeError):
                continue
            node = ctx.graph.nodes[source]
            # Built-in Android/C++ parsers already populate symbol details.
            if node.symbols:
                totals[lang] += len(node.symbols)
                continue
            if lang == "python":
                try:
                    tree = ast.parse(content, filename=source)
                except (SyntaxError, ValueError):
                    continue
                symbols = _python_symbols(ctx, source, tree)
                python_cache[source] = (tree, content.splitlines())
                python_classes[source] = {s["name"] for s in symbols if s["kind"] == "class"}
            elif lang in PATTERNS:
                matches = []
                for kind, pattern in PATTERNS[lang]:
                    for m in re.finditer(pattern, content):
                        line = content.count("\n", 0, m.start()) + 1
                        # Regex heuristic; no fabricated dependency from a symbol match.
                        matches.append({"name":m.group(1), "kind":kind, "fqn":m.group(1),
                                        "line":line, "is_top_level":True, "is_external":False})
                seen = set()
                symbols = []
                for sym in sorted(matches, key=lambda x:(x["line"],x["name"])):
                    key = (sym["line"],sym["name"],sym["kind"])
                    if key not in seen and len(symbols) < 180:
                        seen.add(key)
                        symbols.append(sym)
            else:
                continue
            node.symbols = symbols
            node.search_symbols = [s["name"] for s in symbols][:140]
            totals[lang] += len(symbols)
    inherit = 0
    for source, (tree, lines) in python_cache.items():
        inherit += _python_inheritance(ctx, source, tree, python_classes, lines)
    return {"total_by_language": dict(totals), "resolved_python_inheritance": inherit,
            "note": "Symboly mimo Python jsou zatím heuristické; volání metod nejsou plně indexována."}
