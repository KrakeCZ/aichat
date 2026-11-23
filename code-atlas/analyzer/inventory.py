"""File Inventory and Scope Classifier for Universal Code Atlas.

Discovers and classifies files across the repository, distinguishing
first-party application source, Android resources, native code,
build definitions, and generated build artifacts (e.g. CMake .cxx tree).
"""

from __future__ import annotations

import os
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Inventory-only identification for less common ecosystems. Recognition does NOT
# imply semantic resolution; missing analyzers are explicitly reported.
LANGUAGE_BY_EXTENSION = {
    ".cs":"csharp", ".swift":"swift", ".m":"objective_c", ".mm":"objective_c",
    ".php":"php", ".rb":"ruby", ".sql":"sql", ".r":"r", ".rmd":"r",
    ".jl":"julia", ".scala":"scala", ".sc":"scala", ".groovy":"groovy",
    ".gradle":"groovy", ".dart":"dart", ".lua":"lua", ".pl":"perl",
    ".pm":"perl", ".hs":"haskell", ".lhs":"haskell", ".erl":"erlang",
    ".hrl":"erlang", ".ex":"elixir", ".exs":"elixir", ".clj":"clojure",
    ".cljs":"clojure", ".cljc":"clojure", ".lisp":"lisp", ".lsp":"lisp",
    ".scm":"scheme", ".ss":"scheme", ".rkt":"racket", ".fs":"fsharp",
    ".fsx":"fsharp", ".ml":"ocaml", ".mli":"ocaml", ".sml":"standard_ml",
    ".sig":"standard_ml", ".f":"fortran", ".for":"fortran", ".f90":"fortran",
    ".f95":"fortran", ".f03":"fortran", ".f08":"fortran", ".cob":"cobol",
    ".cbl":"cobol", ".pas":"pascal", ".pp":"pascal", ".dpr":"delphi",
    ".dfm":"delphi", ".adb":"ada", ".ads":"ada", ".pro":"prolog",
    ".st":"smalltalk", ".vb":"visual_basic", ".bas":"visual_basic",
    ".asm":"assembly", ".s":"assembly", ".zig":"zig", ".nim":"nim",
    ".cr":"crystal", ".sol":"solidity", ".vhd":"vhdl", ".vhdl":"vhdl",
    ".v":"verilog", ".sv":"verilog", ".svh":"verilog", ".as":"actionscript",
    ".logo":"logo", ".lgo":"logo", ".mlx":"matlab", ".matlab":"matlab",
}

# A few suffixes are intrinsically ambiguous: .m can be Objective-C or MATLAB,
# .pl Perl or Prolog; .h C, C++ or Objective-C. Heuristic is NOT a compiler.

EXTENSIONS: Set[str] = {
    ".kt", ".java", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".kts", ".py", ".sh", ".ps1", ".xml",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".rs", ".psm1", ".psd1",
    ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".scss", ".properties",
    ".proto", ".graphql", ".gql", ".ini", ".conf", ".md",
    ".csproj", ".fsproj", ".vbproj", ".sln", ".tf"
} | set(LANGUAGE_BY_EXTENSION)

# Folder-name exclusions (not substring matches): avoid vendor/build/secret directories.
EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".cxx", "build", ".gradle", ".idea",
    "dist", "node_modules", ".next", ".nuxt", "coverage", "target",
    ".venv", "venv", "__pycache__", ".pytest_cache", ".code-atlas",
    "atlas-site", ".turbo", ".cache", ".tox", "vendor", "third_party",
})
MANIFEST_NAMES = frozenset({"Cargo.toml", "go.mod", "go.work", "package.json", "tsconfig.json", "jsconfig.json", "pyproject.toml", "composer.json", "Gemfile", "mix.exs", "pubspec.yaml", "Makefile", "Dockerfile", "pom.xml"})
TOOLKIT_PREFIXES: Tuple[str, ...] = (
    "tools/universal-code-atlas/", "tools/code-atlas/", "vendor/universal-code-atlas/",
)


def is_generated_atlas_export(directory: Path) -> bool:
    """Detect an Atlas *output*, regardless of the user's chosen folder name.

    The three independently named artifacts are Atlas' own export signature;
    a normal source tree with a folder named ``symbol-graphs`` alone is NOT
    excluded. Only names/metadata are inspected; no project code is run.
    """
    return (not directory.is_symlink()
            and (directory / 'graph-index.json').is_file()
            and (directory / 'symbol-index.json').is_file()
            and (directory / 'symbol-graphs').is_dir())

# Interpreter names follow the executable names defined by the languages, not
# source-project names. This only identifies file type; it creates no edges.
SHEBANG_LANGUAGES = {
    'python':'python','python2':'python','python3':'python',
    'bash':'shell','sh':'shell','zsh':'shell','ksh':'shell',
    'pwsh':'powershell','powershell':'powershell',
    'node':'javascript','nodejs':'javascript','deno':'javascript','bun':'javascript',
    'ruby':'ruby','perl':'perl','php':'php','lua':'lua','rscript':'r',
}

def language_from_shebang(first_line: str) -> str | None:
    """Recognize a standard hashbang, including /usr/bin/env [-S] program."""
    if not first_line.startswith('#!'):
        return None
    words=first_line[2:].strip().split()
    if not words:
        return None
    executable=words[0].rsplit('/',1)[-1].lower()
    if executable=='env':
        rest=[part for part in words[1:] if not part.startswith('-')]
        if not rest:return None
        executable=rest[0].rsplit('/',1)[-1].lower()
    match=re.fullmatch(r'([a-z]+)(?:[0-9]+(?:\.[0-9]+)?)?',executable)
    return SHEBANG_LANGUAGES.get(executable) or (SHEBANG_LANGUAGES.get(match.group(1)) if match else None)


@dataclass
class FileEntry:
    path: str
    name: str
    extension: str
    language: str
    module: str
    category: str
    size_bytes: int
    is_test: bool
    is_resource: bool
    is_manifest: bool
    is_build_script: bool
    is_generated: bool = False


def determine_category(path: str) -> str:
    """Technology groups instead of specific product-area names."""
    norm = path.replace("\\", "/").lower()
    parts = norm.split("/")
    ext = Path(norm).suffix
    base = parts[-1]
    if any(part in ("tests", "test", "androidtest", "__tests__") for part in parts[:-1]) or base.startswith("test_"):
        return "tests"
    if base in ("cargo.toml", "go.mod", "package.json", "tsconfig.json", "jsconfig.json", "pyproject.toml") or ext == ".psd1" or base.startswith("build.gradle") or base == "cmakelists.txt" or base.startswith("build.gradle") or base.startswith("settings.gradle") or ext in (".cmake", ".kts"):
        return "build"
    if ext in (".cpp", ".cc", ".c", ".h", ".hpp"):
        return "native"
    if ext in (".json", ".toml", ".yaml", ".yml", ".properties", ".ini", ".conf"):
        return "config"
    if ext in (".md",):
        return "documentation"
    if ext in (".html", ".css", ".scss"):
        return "web"
    if ext in (".kt", ".java", ".xml"):
        return "jvm"
    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"):
        return "web"
    if ext in (".py", ".sh", ".ps1", ".psm1", ".psd1"):
        return "scripts"
    if ext in LANGUAGE_BY_EXTENSION:
        return "other_language"
    return "core"


def determine_language(path: str) -> str:
    norm = path.replace("\\", "/")
    basename = Path(norm).name.lower()
    if basename == "cmakelists.txt" or basename.endswith(".cmake"):
        return "cmake"

    ext = Path(norm).suffix.lower()
    if basename == 'cargo.toml': return 'rust_manifest'
    if basename in ('package.json','tsconfig.json','jsconfig.json'): return 'js_config'
    if basename == 'pyproject.toml': return 'python_manifest'
    if basename in ('go.mod','go.work'): return 'go_manifest'
    # .m can mean Objective-C or MATLAB; avoid silently claiming the wrong language.
    # The inexpensive source sniff only determines classification, never an edge.
    if ext == ".m":
        return "ambiguous_m"
    if ext == ".pl":
        return "ambiguous_pl"
    if ext == ".kt":
        return "kotlin"
    if ext == ".kts":
        return "gradle_kts" if basename.startswith(("build.gradle", "settings.gradle")) else "kotlin_script"
    if ext == ".java":
        return "java"
    if ext in (".cpp", ".cc"):
        return "cpp"
    if ext == ".c":
        return "c"
    if ext in (".h", ".hpp"):
        return "header"
    if ext == ".xml":
        return "xml"
    if ext in (".json", ".toml", ".yaml", ".yml", ".html", ".css", ".scss", ".properties", ".proto", ".graphql", ".gql", ".ini", ".conf", ".md", ".tf", ".csproj", ".fsproj", ".vbproj", ".sln"):
        return ext.lstrip(".")
    if ext == ".py":
        return "python"
    if ext == ".sh":
        return "shell"
    if ext in (".ps1", ".psm1", ".psd1"):
        return "powershell"
    if ext in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    if ext in (".ts", ".tsx", ".mts", ".cts"):
        return "typescript"
    if ext == ".go":
        return "go"
    if ext == ".rs":
        return "rust"
    if basename in ("tsconfig.json", "jsconfig.json", "package.json"):
        return "js_config"
    if basename == "cargo.toml":
        return "rust_manifest"
    if basename == "go.mod":
        return "go_manifest"
    if basename == "pyproject.toml":
        return "python_manifest"
    return LANGUAGE_BY_EXTENSION.get(ext, "unknown")


def determine_module(path: str) -> str:
    """Heuristic module label; never assume project-specific module names."""
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if len(parts) > 1:
        # Prefer an Android/Gradle module near src/, otherwise the top directory.
        if "src" in parts:
            idx = parts.index("src")
            return f":{parts[idx-1]}" if idx > 0 else ":root"
        return ":" + parts[0]
    return ":root"


def is_included_path(path: str, prefix: str = "") -> bool:
    norm = path.replace("\\", "/")
    if prefix and not norm.startswith(prefix):
        return False

    if any(part in EXCLUDED_DIRS for part in norm.split("/")[:-1]):
        return False

    basename = Path(norm).name
    ext = Path(norm).suffix.lower()
    return ext in EXTENSIONS or basename in MANIFEST_NAMES or basename == "CMakeLists.txt" or basename.endswith(".cmake")


class Inventory:
    def __init__(self, root_dir: Path, prefix: str = ""):
        self.root_dir = Path(root_dir).resolve()
        self.prefix = prefix
        self.files: Dict[str, FileEntry] = {}
        self.generated_artifacts: Dict[str, FileEntry] = {}
        self.excluded_atlas_exports: List[str] = []
        self.omitted_large_sources: List[dict] = []
        self.omitted_large_source_count = 0
        self.source_commit: str = ""

    def scan(self) -> List[FileEntry]:
        self.files.clear()
        self.generated_artifacts.clear()
        self.excluded_atlas_exports.clear()
        self.omitted_large_sources.clear()
        self.omitted_large_source_count = 0
        paths: List[str] = []

        # Filesystem scan includes new/untracked project files. Git ls-files alone
        # silently omitted them. Never follow symlinked directories or files.
        self.source_commit = "local"
        try:
            self.source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root_dir,
                stderr=subprocess.DEVNULL, timeout=3
            ).decode().strip()
        except (OSError, subprocess.SubprocessError):
            pass
        for walk_root, dirs, filenames in os.walk(self.root_dir):
            # Atlas output is not project source.  A user can name the output
            # directory anything (e.g. test2 or artifacts/v1), so excluding
            # fixed names alone incorrectly re-indexes thousands of old
            # symbol-graphs/*.json files in the new project graph.
            if is_generated_atlas_export(Path(walk_root)):
                self.excluded_atlas_exports.append(
                    Path(walk_root).relative_to(self.root_dir).as_posix())
                dirs[:] = []
                continue
            included_dirs = []
            for name in dirs:
                candidate = Path(walk_root) / name
                if name in EXCLUDED_DIRS or candidate.is_symlink():
                    continue
                if is_generated_atlas_export(candidate):
                    self.excluded_atlas_exports.append(
                        candidate.relative_to(self.root_dir).as_posix())
                    continue
                included_dirs.append(name)
            dirs[:] = included_dirs
            for name in filenames:
                absolute = Path(walk_root) / name
                if not absolute.is_symlink():
                    paths.append(absolute.relative_to(self.root_dir).as_posix())

        for p in sorted(paths):
            norm = p.replace("\\", "/")
            if self.prefix and not norm.startswith(self.prefix):
                continue

            basename = Path(norm).name
            ext = Path(norm).suffix.lower()
            is_valid_type = ext in EXTENSIONS or basename in MANIFEST_NAMES or basename == "CMakeLists.txt" or basename.endswith(".cmake")
            # Extensionless scripts are common in polyglot repositories. Only
            # inspect a tiny first line and never execute the file.
            shebang_language = None
            if not is_valid_type and not ext:
                try:
                    with (self.root_dir / p).open('rb') as handle:
                        shebang_language = language_from_shebang(handle.readline(256).decode('utf-8',errors='replace'))
                except OSError:
                    pass
                is_valid_type = bool(shebang_language)
            if not is_valid_type:
                continue

            # Exclude generated and third-party directories from analysis.
            # This handles both git-index and filesystem fallback scans.
            if any(part in EXCLUDED_DIRS for part in norm.split("/")[:-1]):
                continue
            if norm.startswith(TOOLKIT_PREFIXES):
                continue
            is_gen = False

            abs_path = self.root_dir / p
            if not abs_path.is_file() or abs_path.is_symlink():
                continue

            size = abs_path.stat().st_size
            if size > 2_000_000:
                self.omitted_large_source_count += 1
                if len(self.omitted_large_sources) < 10_000:
                    self.omitted_large_sources.append({
                        'path':p,'bytes':size,'reason':'source_exceeds_inventory_limit'})
                continue
            lang = shebang_language or determine_language(p)
            if lang in ("ambiguous_m", "ambiguous_pl"):
                try:
                    head = abs_path.open("rb").read(12000).decode("utf-8", errors="replace")
                except OSError:
                    head = ""
                if lang == "ambiguous_m":
                    if "@implementation" in head or "@interface" in head or "#import" in head:
                        lang = "objective_c"
                    elif "classdef" in head or "function " in head or "%%" in head:
                        lang = "matlab"
                elif lang == "ambiguous_pl":
                    if "use strict" in head or "use warnings" in head or "sub " in head:
                        lang = "perl"
                    elif ":-" in head:
                        lang = "prolog"
            mod = determine_module(p)
            cat = determine_category(p)
            is_test = "/test/" in p or "/androidTest/" in p or "/tests/" in p
            is_res = "/res/" in p
            is_manifest = basename == "AndroidManifest.xml"
            is_build = ext == ".kts" or basename in MANIFEST_NAMES or basename == "CMakeLists.txt" or basename.endswith(".cmake")

            entry = FileEntry(
                path=p,
                name=basename,
                extension=ext,
                language=lang,
                module=mod,
                category=cat,
                size_bytes=size,
                is_test=is_test,
                is_resource=is_res,
                is_manifest=is_manifest,
                is_build_script=is_build,
                is_generated=is_gen,
            )

            if is_gen:
                self.generated_artifacts[p] = entry
            else:
                self.files[p] = entry

        return list(self.files.values())

    def get_summary(self) -> Dict[str, Any]:
        return {
            "primary_source_count": len(self.files),
            "generated_artifact_count": len(self.generated_artifacts),
            "omitted_large_source_count": self.omitted_large_source_count,
            "omitted_large_sources": list(self.omitted_large_sources),
            "excluded_atlas_export_roots": sorted(set(self.excluded_atlas_exports)),
            "source_commit": self.source_commit,
            "languages": {
                lang: sum(1 for f in self.files.values() if f.language == lang)
                for lang in sorted({f.language for f in self.files.values()})
            },
            "modules": {
                mod: sum(1 for f in self.files.values() if f.module == mod)
                for mod in sorted({f.module for f in self.files.values()})
            },
        }
