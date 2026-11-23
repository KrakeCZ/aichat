"""Rust local `mod`, `use crate/self/super`, #[path], include! paths."""
from __future__ import annotations
import posixpath
import re
from dataclasses import dataclass
from . import Context
from analyzer.model import EdgeType, ConfidenceLevel

MOD = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;', re.M)
USE = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);', re.M)
PATH_ATTR = re.compile(r'#\s*\[\s*path\s*=\s*"([^"]+)"\s*\]')
INCLUDE = re.compile(r'\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"\s*\)')


def crate_for(path: str, cargos: list[str]) -> str:
    parents = [posixpath.dirname(c) for c in cargos]
    candidates = [p for p in parents if not p or path.startswith(p + "/")]
    if candidates:
        return max(candidates, key=len)
    # Projects may consist of loose Rust files without a Cargo.toml;
    # infer the local src root but never infer external crate dependencies.
    if "/src/" in path:
        return path.split("/src/",1)[0]
    return ""


def module_path(path: str, crate: str) -> list[str]:
    rel = path[len(crate):].lstrip("/") if crate else path
    rel = rel.removeprefix("src/")
    if rel in ("lib.rs", "main.rs", "mod.rs"):
        return []
    chunks = rel.removesuffix(".rs").split("/")
    if chunks[-1] == "mod":
        chunks.pop()
    return chunks


@dataclass
class RustPlugin:
    name: str = "rust"
    languages: tuple[str, ...] = ("rust",)

    def analyze(self, ctx: Context) -> None:
        cargos = sorted(p for p in ctx.paths if p.endswith("Cargo.toml"))
        rust_files = ctx.files_by_language.get("rust", [])
        maps: dict[str, dict[tuple[str, ...], str]] = {}
        for source in rust_files:
            crate = crate_for(source, cargos)
            maps.setdefault(crate, {})[tuple(module_path(source, crate))] = source
        for source in rust_files:
            crate = crate_for(source, cargos)
            mods = maps[crate]
            location = module_path(source, crate)
            directory = posixpath.dirname(source)
            content = ctx.read(source)
            lines = content.splitlines()
            for match in MOD.finditer(content):
                name = match.group(1)
                line = content.count("\n", 0, match.start()) + 1
                preceding = content[:match.start()].splitlines()[-2:]
                attr = PATH_ATTR.search("\n".join(preceding))
                if attr:
                    path = posixpath.normpath(posixpath.join(directory, attr.group(1)))
                    target = path if path in ctx.paths else None
                else:
                    target = mods.get(tuple(location + [name]))
                if target and target != source:
                    ctx.add(source, target, line, lines[line-1], self.name, EdgeType.IMPORT,
                            ConfidenceLevel.RESOLVED_STATIC)
                else:
                    ctx.miss(self.name, source, line, f"mod {name}")
            for match in USE.finditer(content):
                ref = re.sub(r'\s+', '', match.group(1)).lstrip(':')
                line = content.count("\n", 0, match.start()) + 1
                path_bits = ref.split("::")
                if path_bits[0] == "crate":
                    bits = []
                    path_bits.pop(0)
                elif path_bits[0] == "self":
                    bits = location[:]
                    path_bits.pop(0)
                elif path_bits[0] == "super":
                    bits = location[:]
                    while path_bits and path_bits[0] == "super":
                        if bits:
                            bits.pop()
                        path_bits.pop(0)
                else:
                    bits = []  # Rust 2018 use paths are absolute at crate root (or extern crates)
                # Find longest matching local module prefix: use crate::foo::Bar
                target = None
                for part in path_bits:
                    if not re.fullmatch(r'\w+', part):
                        break
                    bits.append(part)
                    target = mods.get(tuple(bits)) or target
                if target and target != source:
                    ctx.add(source, target, line, lines[line-1], self.name, EdgeType.IMPORT)
            for match in INCLUDE.finditer(content):
                path = posixpath.normpath(posixpath.join(directory, match.group(1)))
                if path in ctx.paths and path != source:
                    line = content.count("\n", 0, match.start()) + 1
                    ctx.add(source, path, line, lines[line-1], self.name, EdgeType.INCLUDE)


PLUGIN = RustPlugin()
