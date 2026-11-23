"""Evidence-first inside-file graphs, initially with a Python AST semantic adapter.

This module is deliberately distinct from the V0 file-relationship engine.
No analyzed program is imported or executed. Cross-file calls are emitted ONLY
when a local Python module and a unique top-level declaration are resolved.
Other languages receive an honest declaration-only preview from existing indexers.
"""
from __future__ import annotations

import ast
import hashlib
import io
import tokenize
import token
import json
import posixpath
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "atlas-symbol-graph/0.1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _node_id(path: str, qualname: str, line: int, col: int) -> str:
    return "symbol:" + _sha(f"{path}\0{qualname}\0{line}\0{col}".encode())[:24]


def _file_id(path: str) -> str:
    return "file:" + path


def _is_name_target(target: ast.AST) -> list[ast.Name]:
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _is_name_target(elt)]
    return []


def _statements(body: list[ast.stmt]):
    """Statements of a lexical scope; never descend into child scopes."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field, value in ast.iter_fields(stmt):
            if field in ("body", "orelse", "finalbody", "handlers", "cases"):
                if isinstance(value, list):
                    for child in value:
                        if isinstance(child, ast.stmt):
                            yield from _statements([child])
                        elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                            yield from _statements(child.body)


def _walk_scope(node: ast.AST):
    """Traverse expressions and statements, excluding nested code scopes."""
    stack = [node]
    while stack:
        item = stack.pop()
        yield item
        if item is not node and isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(item))))


def _module_target(source: str, module: str, level: int, known: set[str]) -> str | None:
    parent = posixpath.dirname(source)
    if level:
        base = parent
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        candidates = [posixpath.join(base, module.replace(".", "/"))]
    else:
        name = module.replace(".", "/")
        candidates = [name]
        parts = source.split("/")
        for i, part in enumerate(parts[:-1]):
            if part in ("src", "python", "lib"):
                candidates.append(posixpath.join("/".join(parts[:i+1]), name))
        # Never assume the caller's package directory is on sys.path.
        # In `pkg/consumer.py`, plain `import helper` is absolute and does NOT
        # mean `pkg/helper.py`; that would require `from . import helper`.
    matches = set()
    for path in candidates:
        base = posixpath.normpath(path)
        if base == ".." or base.startswith("../") or base.startswith("/"):
            continue
        for opt in (base + ".py", base.rstrip("/") + "/__init__.py"):
            if opt in known:
                matches.add(opt)
    return next(iter(matches)) if len(matches) == 1 else None


class PythonFile:
    def __init__(self, path: str, source: str, source_hash: str | None = None):
        self.path = path
        self.source = source
        self.lines = source.splitlines()
        self.hash = source_hash or _sha(source.encode("utf-8"))
        self.tree = ast.parse(source, filename=path, type_comments=True)
        self.python_source_syntax = self._source_syntax_index()
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.by_ast: dict[int, dict] = {}
        self.top: dict[str, list[dict]] = defaultdict(list)
        self.scope_declarations: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self.scopes: list[tuple[ast.AST, dict, list[ast.stmt]]] = []
        self._edge_set: set[tuple] = set()
        self.file_node = {"id": _file_id(path), "path": path, "name": path.split("/")[-1],
                          "kind": "file", "line": 1, "end_line": len(self.lines),
                          "confidence": "verified_syntax"}
        self.nodes.append(self.file_node)
        self._collect(self.tree.body, self.file_node, "")
        self.scopes.insert(0, (self.tree, self.file_node, self.tree.body))

    def _source_syntax_index(self) -> dict:
        """Read-only lossless *coordinates* for Python AST + lexical tokens.

        Full source lexemes are intentionally not duplicated into a JSON export:
        `--bundle-sources` remains the user's explicit source distribution
        choice. AST positions use UTF-8 byte columns, tokenize columns are
        Unicode code point positions, as defined by their respective APIs.
        """
        limit = 150_000
        ast_nodes = []
        stack = [(self.tree, None, None)]
        while stack and len(ast_nodes) < limit:
            node, parent, field = stack.pop()
            idx = len(ast_nodes)
            row = {'id': idx, 'type': type(node).__name__, 'parent': parent,
                   'field': field}
            if hasattr(node, 'lineno'):
                # Not every official CPython AST node with a line number has
                # column offsets. In particular, ``ast.TypeIgnore`` is a
                # line-only annotation produced by ``type_comments=True``.
                # Never invent an exact column or crash the entire adapter.
                row['line'] = int(node.lineno)
                if getattr(node, 'end_lineno', None) is not None:
                    row['end_line'] = int(node.end_lineno)
                if getattr(node, 'col_offset', None) is not None:
                    row['column_utf8'] = int(node.col_offset)
                if getattr(node, 'end_col_offset', None) is not None:
                    row['end_column_utf8'] = int(node.end_col_offset)
                if 'column_utf8' not in row:
                    row['position_precision'] = 'line_only'
            # Names and type signatures are source declarations, not dynamic
            # runtime values. Literal payloads are not replicated.
            if isinstance(node, ast.Name):
                row['identifier'] = node.id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                row['identifier'] = node.name
            elif isinstance(node, ast.Attribute):
                row['identifier'] = node.attr
            elif isinstance(node, ast.arg):
                row['identifier'] = node.arg
            elif isinstance(node, ast.Constant):
                row['literal_kind'] = type(node.value).__name__
            ast_nodes.append(row)
            children = []
            for key, value in ast.iter_fields(node):
                if isinstance(value, ast.AST):
                    children.append((value, idx, key))
                elif isinstance(value, list):
                    children.extend((item, idx, f'{key}[{i}]')
                                    for i, item in enumerate(value) if isinstance(item, ast.AST))
            stack.extend(reversed(children))
        tokens = []
        lex_error = None
        try:
            for tok in tokenize.generate_tokens(io.StringIO(self.source).readline):
                if len(tokens) >= limit:
                    lex_error = 'token_limit_exceeded'
                    break
                tokens.append({'kind':token.tok_name.get(tok.type, str(tok.type)),
                               'line':tok.start[0], 'column_codepoints':tok.start[1],
                               'end_line':tok.end[0], 'end_column_codepoints':tok.end[1]})
        except (tokenize.TokenError, IndentationError, SyntaxError, ValueError) as exc:
            lex_error = type(exc).__name__
        complete = not stack and lex_error is None
        return {'schema':'atlas-python-source-syntax/1',
                'source_sha256':self.hash,
                'ast_column_unit':'utf8_bytes',
                'token_column_unit':'unicode_codepoints',
                'status':'complete_source_evidence' if complete else 'bounded_or_lexical_error',
                'ast_nodes':ast_nodes, 'tokens':tokens,
                'limitations':([lex_error] if lex_error else []) +
                               (['ast_node_limit_exceeded'] if stack else []),
                'note':'AST structural topology and lexical spans, not a full concrete syntax tree or semantic proof.'}

    def _source_evidence(self, node: ast.AST) -> dict:
        line = int(getattr(node, "lineno", 1))
        return {"path": self.path, "line": line,
                "end_line": int(getattr(node, "end_lineno", line)),
                "column": int(getattr(node, "col_offset", 0)),
                "end_column": int(getattr(node, "end_col_offset", getattr(node, "col_offset", 0))),
                "source_sha256": self.hash,
                "snippet": (self.lines[line-1].strip()[:200] if 0 < line <= len(self.lines) else "")}

    def edge(self, source: dict, target: dict, relation: str, at: ast.AST,
             resolution: str = "verified_syntax") -> None:
        key = (source["id"], target["id"], relation,
               int(getattr(at, "lineno", 1)), int(getattr(at, "col_offset", 0)),
               int(getattr(at, "end_lineno", 0)), int(getattr(at, "end_col_offset", 0)))
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        self.edges.append({"source": source["id"], "target": target["id"],
                           "source_path": source["path"], "target_path": target["path"],
                           "relation": relation, "resolution": resolution,
                           "analyzer": "python-ast", "evidence": self._source_evidence(at)})

    def _create(self, astnode: ast.AST, parent: dict, qualname: str, kind: str, name: str) -> dict:
        line, col = int(getattr(astnode, "lineno", 1)), int(getattr(astnode, "col_offset", 0))
        obj = {"id": _node_id(self.path, qualname, line, col), "path": self.path,
               "name": name, "qualname": qualname, "kind": kind,
               "line": line, "end_line": int(getattr(astnode, "end_lineno", line)),
               "column": col, "end_column": int(getattr(astnode, "end_col_offset", col)),
               "parent": parent["id"], "confidence": "verified_syntax"}
        self.nodes.append(obj)
        self.edge(parent, obj, "declares", astnode)
        self.by_ast[id(astnode)] = obj
        self.scope_declarations[parent["id"]][name].append(obj)
        if parent["kind"] == "file" and kind in ("function", "class"):
            self.top[name].append(obj)
        return obj

    def _collect(self, body: list[ast.stmt], parent: dict, prefix: str) -> None:
        for stmt in _statements(body):
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "class" if isinstance(stmt, ast.ClassDef) else ("async_function" if isinstance(stmt, ast.AsyncFunctionDef) else "function")
                if parent["kind"] == "class" and kind != "class":
                    kind = "method"
                qual = (prefix + "." if prefix else "") + stmt.name
                obj = self._create(stmt, parent, qual, kind, stmt.name)
                obj["conditional_declaration"] = all(stmt is not direct for direct in body)
                obj["decorated"] = bool(stmt.decorator_list)
                if isinstance(stmt, ast.ClassDef):
                    obj["custom_metaclass"] = any(k.arg == "metaclass" for k in stmt.keywords)
                self.scopes.append((stmt, obj, stmt.body))
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    obj['returns_annotation'] = (ast.get_source_segment(self.source, stmt.returns)
                                                  if stmt.returns is not None else None)
                    obj['declared_arity'] = {
                        'positional': len(stmt.args.posonlyargs) + len(stmt.args.args),
                        'positional_only': len(stmt.args.posonlyargs),
                        'keyword_only': len(stmt.args.kwonlyargs),
                        'defaults': len(stmt.args.defaults),
                        'vararg': stmt.args.vararg.arg if stmt.args.vararg else None,
                        'kwarg': stmt.args.kwarg.arg if stmt.args.kwarg else None,
                    }
                    args = stmt.args.posonlyargs + stmt.args.args + stmt.args.kwonlyargs
                    if stmt.args.vararg: args.append(stmt.args.vararg)
                    if stmt.args.kwarg: args.append(stmt.args.kwarg)
                    for arg in args:
                        param = self._create(arg, obj, qual + ".param:" + arg.arg, "parameter", arg.arg)
                        param['annotation'] = (ast.get_source_segment(self.source, arg.annotation)
                                               if arg.annotation is not None else None)
                self._collect(stmt.body, obj, qual)
            elif hasattr(ast, 'TypeAlias') and isinstance(stmt, ast.TypeAlias):
                if isinstance(stmt.name, ast.Name):
                    qual = (prefix + '.' if prefix else '') + stmt.name.id
                    self._create(stmt.name, parent, qual, 'type_alias', stmt.name.id)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                # Only simple named bindings defined in the current lexical scope.
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    for name in _is_name_target(target):
                        kind = "field" if parent["kind"] == "class" else "variable"
                        if parent["kind"] == "file" or parent["kind"] == "class":
                            self._create(name, parent, (prefix + "." if prefix else "") + name.id, kind, name.id)
                # Function-local variables are indexed in the next pass where scope is known.

    def new_local(self, expr: ast.Name, owner: dict) -> dict:
        scope = owner.get("qualname", "")
        qual = scope + ".local:" + expr.id
        return self._create(expr, owner, qual, "local_variable", expr.id)


def _unique(nodes: list[dict] | None) -> dict | None:
    return nodes[0] if nodes and len(nodes) == 1 else None


@dataclass(frozen=True)
class _PythonImport:
    path: str
    symbol: str | None
    # `import pkg.sub` binds `pkg`, not `pkg.sub`. The intermediate
    # attributes must actually be present at the call site.
    prefix: tuple[str, ...] = ()
    unconditional: bool = True


def _python_import_bindings(path: str, tree: ast.Module,
                            parsed: dict[str, PythonFile]) -> dict[str, _PythonImport]:
    known = set(parsed)
    bindings: dict[str, _PythonImport] = {}
    conflicted: set[str] = set()
    # An import under if/try/for/with may not execute: never treat it as a
    # guaranteed lexical binding. Import-star is deliberately unsupported.
    for stmt in _statements(tree.body):
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        direct = any(stmt is item for item in tree.body)
        for alias in stmt.names:
            if alias.name == '*':
                conflicted.add('*')
                continue
            if isinstance(stmt, ast.Import):
                bound = alias.asname or alias.name.split('.')[0]
                target = _module_target(path, alias.name, 0, known)
                if not target:
                    conflicted.add(bound)
                    continue
                ref = _PythonImport(target, None,
                                    () if alias.asname else tuple(alias.name.split('.')[1:]), direct)
            else:
                bound = alias.asname or alias.name
                base = _module_target(path, stmt.module or '', stmt.level, known)
                # A from-import can denote an exported declaration or a package
                # submodule. Do not choose a declaration by name alone.
                direct_symbol = base and _unique(parsed[base].top.get(alias.name))
                # A package __init__.py may explicitly re-export a symbol:
                # `from .worker import run` followed by `from pkg import run`.
                # Only source-visible direct import declarations qualify;
                # the recursive resolver rejects cycles and ambiguity.
                explicit_reexport = bool(base and any(
                    isinstance(item, (ast.Import, ast.ImportFrom)) and any(
                        (a.asname or (a.name.split('.')[0] if isinstance(item, ast.Import) else a.name)) == alias.name
                        for a in item.names)
                    for item in parsed[base].tree.body))
                if direct_symbol is not None or explicit_reexport:
                    ref = _PythonImport(base, alias.name, (), direct)
                else:
                    sub = _module_target(path,
                        ((stmt.module + '.') if stmt.module else '') + alias.name,
                        stmt.level, known)
                    if not sub:
                        conflicted.add(bound)
                        continue
                    ref = _PythonImport(sub, None, (), direct)
            if bound in bindings and bindings[bound] != ref:
                conflicted.add(bound)
            bindings[bound] = ref
    return {name: ref for name, ref in bindings.items()
            if name not in conflicted and '*' not in conflicted and ref.unconditional}


def _owner_shadow_names(st: ast.AST) -> set[str]:
    """Names whose binding may differ from the project-global declaration.

    A function-local assignment shadows a global throughout the *entire*
    function, even before the assignment. We deliberately do not guess which
    runtime branch executes or infer function values from assignments.
    """
    shadows: set[str] = set()
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = st.args.posonlyargs + st.args.args + st.args.kwonlyargs
        shadows.update(a.arg for a in args)
        if st.args.vararg:
            shadows.add(st.args.vararg.arg)
        if st.args.kwarg:
            shadows.add(st.args.kwarg.arg)
    for n in _walk_scope(st):
        if n is not st and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadows.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            shadows.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            shadows.update(a.asname or a.name.split('.')[0] for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            shadows.add(n.name)
        elif isinstance(n, ast.MatchAs) and n.name:
            shadows.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            shadows.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            shadows.add(n.rest)
    return shadows


def _callable_decl(node: dict | None) -> bool:
    return bool(node and node['kind'] in ('function', 'async_function', 'method', 'class')
                and not node.get('conditional_declaration')
                and not node.get('decorated')
                and not node.get('custom_metaclass'))


def _name_and_attributes(expr: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    suffix = []
    while isinstance(expr, ast.Attribute):
        suffix.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        return expr.id, tuple(reversed(suffix))
    return None


def _analyze_python_references(parsed: dict[str, PythonFile]) -> None:
    """Conservative project-wide AST resolver; never imports analyzed modules.

    Exact cross-file calls require a unique source declaration and an
    unambiguous import spelling. Runtime patches, dynamic attributes,
    conditional imports/declarations and decorators are never promoted.
    """
    aliases = {p: _python_import_bindings(p, pf.tree, parsed) for p, pf in parsed.items()}
    module_import_conflicts = {
        path: {a.asname or (a.name.split('.')[0] if isinstance(item, ast.Import) else a.name)
               for item in _statements(pf.tree.body)
               if isinstance(item, (ast.Import, ast.ImportFrom))
               for a in item.names}
        for path, pf in parsed.items()
    }
    module_star_imports = {
        path for path, pf in parsed.items()
        if any(isinstance(item, ast.ImportFrom) and any(a.name == '*' for a in item.names)
               for item in _statements(pf.tree.body))
    }
    # An import may be explicitly re-exported by another module. Follow only
    # unique, local, source-known import chains; detect cycles.
    def exported(path: str, symbol: str, visited: frozenset[tuple[str, str]] = frozenset()):
        key = (path, symbol)
        if key in visited:
            return None
        pf = parsed[path]
        if path in module_star_imports:
            return None
        decl = _unique(pf.top.get(symbol))
        if _callable_decl(decl) and symbol not in module_import_conflicts[path]:
            return decl
        if symbol in pf.scope_declarations.get(pf.file_node['id'], {}):
            return None
        ref = aliases[path].get(symbol)
        if ref and ref.symbol and not ref.prefix:
            return exported(ref.path, ref.symbol, visited | {key})
        return None

    for path, pf in parsed.items():
        imports = aliases[path]
        nodes_by_id = {n['id']: n for n in pf.nodes}
        global_decl = pf.scope_declarations.get(pf.file_node['id'], {})
        # Any module-level assignment, deletion or multiple definitions can
        # change a binding. Fail closed even if a later branch seems unlikely.
        module_shadow = _owner_shadow_names(pf.tree)
        module_rebound = {name for name, entries in global_decl.items()
                          if any(n['kind'] in ('variable', 'field') for n in entries)}
        module_decl_names = set(global_decl)
        # Nonlocal rebinding can replace a closure function even when the
        # mutating statement is in a different nested function.
        nonlocal_rebound = {name for item in ast.walk(pf.tree)
                            if isinstance(item, ast.Nonlocal) for name in item.names}
        # Name binding may be changed by a module-level `global` (not meaningful)
        # or by assignments in subscopes using `global` declarations.
        for astscope, _, _ in pf.scopes:
            if astscope is pf.tree:
                continue
            globals_declared = {name for n in ast.walk(astscope)
                                if isinstance(n, ast.Global) for name in n.names}
            if globals_declared:
                module_rebound.update(globals_declared)
        # Direct import edges are *declaration references*, not calls.
        for st in pf.tree.body:
            if not isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            for a in st.names:
                bound = a.asname or (a.name.split('.')[0] if isinstance(st, ast.Import) else a.name)
                ref = imports.get(bound)
                if ref:
                    target_file = parsed[ref.path].file_node
                    pf.edge(pf.file_node, target_file, 'imports', st, 'resolved_declaration')
        reassigned_methods: dict[str, set[str]] = defaultdict(set)
        for scope_ast, owner, _ in pf.scopes:
            parent_node = nodes_by_id.get(owner.get('parent'))
            if not parent_node or parent_node['kind'] != 'class':
                continue
            for node in _walk_scope(scope_ast):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    if isinstance(node.value, ast.Name):
                        reassigned_methods[parent_node['id']].add(node.attr)

        scope_shadows = {owner['id']: _owner_shadow_names(astscope)
                         for astscope, owner, _ in pf.scopes}
        scope_writes = {owner['id']: {n.id for n in _walk_scope(astscope)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
                         for astscope, owner, _ in pf.scopes}
        scope_by_id = {owner['id']: (astscope, owner, body)
                       for astscope, owner, body in pf.scopes}
        # Imports in function bodies are lexical too, but can only be used as
        # exact module references when they are unconditional source statements.
        local_imports: dict[str, dict[str, _PythonImport]] = {}
        for astscope, owner, body in pf.scopes:
            if astscope is pf.tree:
                continue
            import_tree = ast.Module(body=body, type_ignores=[])
            local_imports[owner['id']] = _python_import_bindings(path, import_tree, parsed)

        def resolve_ref(ref: _PythonImport, attrs: tuple[str, ...]):
            if ref.prefix:
                if attrs[:len(ref.prefix)] != ref.prefix:
                    return None
                attrs = attrs[len(ref.prefix):]
            if ref.symbol is not None:
                return exported(ref.path, ref.symbol) if not attrs else None
            if len(attrs) != 1:
                return None
            return exported(ref.path, attrs[0])

        def resolved_name(name: str, scope_id: str, attrs: tuple[str, ...]):
            """Statically resolve a lexical declaration or import, never values."""
            seen: set[str] = set()
            current = scope_id
            while current and current not in seen:
                seen.add(current)
                astscope, owner, _ = scope_by_id[current]
                is_file = owner['kind'] == 'file'
                decls = pf.scope_declarations.get(current, {}).get(name)
                candidate = _unique(decls)
                shadows = scope_shadows[current]
                # A store/del in the scope invalidates a local function,
                # import, and an enclosing binding for the whole scope.
                shadowed = name in shadows
                if candidate and not attrs and _callable_decl(candidate):
                    if (name not in scope_writes[current]
                            and name not in nonlocal_rebound
                            and (not is_file or (name not in module_rebound
                                                  and name not in module_import_conflicts[path]
                                                  and path not in module_star_imports))):
                        return candidate
                if decls or (shadowed and not is_file):
                    # A directly imported local name is special: `_owner_shadow_names`
                    # marks all imports as bindings. It is accepted only if
                    # there is no other store/del to that name.
                    ref = local_imports.get(current, {}).get(name)
                    if ref:
                        local_stores = [n for n in _walk_scope(astscope)
                                        if isinstance(n, ast.Name) and n.id == name
                                        and isinstance(n.ctx, (ast.Store, ast.Del))]
                        if not local_stores:
                            return resolve_ref(ref, attrs)
                    return None
                ref = (imports if is_file else local_imports.get(current, {})).get(name)
                if ref:
                    if is_file and (name in module_rebound or name in module_decl_names
                                    or path in module_star_imports):
                        return None
                    if not is_file:
                        local_stores = [n for n in _walk_scope(astscope)
                                        if isinstance(n, ast.Name) and n.id == name
                                        and isinstance(n.ctx, (ast.Store, ast.Del))]
                        if local_stores:
                            return None
                    return resolve_ref(ref, attrs)
                if is_file:
                    return None
                parent_id = owner.get('parent')
                parent = nodes_by_id.get(parent_id)
                # Python functions skip class scopes for unqualified lookups.
                while parent and parent['kind'] == 'class' and owner['kind'] != 'class':
                    parent_id = parent.get('parent')
                    parent = nodes_by_id.get(parent_id)
                if parent_id not in scope_by_id:
                    break
                current = parent_id
            return None

        def walk_owned(node: ast.AST):
            """Yield only nodes evaluated in this scope (not nested bodies)."""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Decorators, default expressions and annotations run in the
                # containing scope, not the function's runtime body.
                for child in [*node.decorator_list, *node.args.defaults,
                              *(x for x in node.args.kw_defaults if x is not None)]:
                    yield from walk_owned(child)
                return
            if isinstance(node, ast.ClassDef):
                for child in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
                    yield from walk_owned(child)
                return
            if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Lambda/comprehension bodies have their own local bindings.
                # They are not incorrectly charged to the enclosing function.
                return
            yield node
            for child in ast.iter_child_nodes(node):
                yield from walk_owned(child)

        def declaration_for_read(name: str, scope_id: str):
            """A source-level lexical declaration, not a runtime value target."""
            current = scope_id
            visited = set()
            while current in scope_by_id and current not in visited:
                visited.add(current)
                astscope, owner, _ = scope_by_id[current]
                decls = pf.scope_declarations.get(current, {}).get(name)
                decl = _unique(decls)
                if decl:
                    return decl
                if name in scope_shadows[current] and owner['kind'] != 'file':
                    return None
                parent_id = owner.get('parent')
                parent = nodes_by_id.get(parent_id)
                while parent and parent['kind'] == 'class' and owner['kind'] != 'class':
                    parent_id = parent.get('parent')
                    parent = nodes_by_id.get(parent_id)
                current = parent_id
            return None

        for astscope, owner, body in pf.scopes:
            scope = owner['id']
            declared_globals = {n for st in body for n in (
                name for item in _walk_scope(st) if isinstance(item, ast.Global)
                for name in item.names)}
            declared_nonlocals = {n for st in body for n in (
                name for item in _walk_scope(st) if isinstance(item, ast.Nonlocal)
                for name in item.names)}
            parent_node = nodes_by_id.get(owner.get('parent'))
            enclosing_class = parent_node if parent_node and parent_node['kind'] == 'class' else None
            # Walk statements *once*: previously overlapping subtrees could
            # conflate calls on the same line and produce extra writes.
            for stmt in body:
                for node in walk_owned(stmt):
                    if isinstance(node, ast.Call):
                        expr = _name_and_attributes(node.func)
                        if expr is None:
                            continue
                        name, attrs = expr
                        is_receiver = False
                        target = None
                        if attrs and enclosing_class and isinstance(astscope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            params = astscope.args.posonlyargs + astscope.args.args
                            is_receiver = bool(params and params[0].arg == name and len(attrs) == 1)
                            if is_receiver and attrs[0] not in reassigned_methods[enclosing_class['id']]:
                                target = _unique(pf.scope_declarations.get(enclosing_class['id'], {}).get(attrs[0]))
                                if target and not _callable_decl(target):
                                    target = None
                        if target is None and not is_receiver:
                            target = resolved_name(name, scope, attrs)
                        if target:
                            relation = 'instantiates' if target['kind'] == 'class' else 'calls'
                            resolution = 'candidate_dispatch' if is_receiver else 'resolved_static'
                            pf.edge(owner, target, relation, node, resolution)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        if owner['kind'] in ('function','async_function','method'):
                            # Explicit global/nonlocal writes are NOT new local
                            # variables. Resolve to an existing lexical binding
                            # or withhold a false declaration.
                            if node.id in declared_globals:
                                target = _unique(global_decl.get(node.id))
                            elif node.id in declared_nonlocals:
                                parent_scope = owner.get('parent')
                                target = (declaration_for_read(node.id, parent_scope)
                                          if parent_scope else None)
                            else:
                                declarations = pf.scope_declarations.get(scope, {}).get(node.id)
                                target = _unique(declarations)
                                if target is None:
                                    target = pf.new_local(node, owner)
                            if target:
                                pf.edge(owner, target, 'writes', node, 'resolved_declaration')
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        # Reads are lexical uses, not a claim about the actual
                        # runtime object or branch execution. A callable cross-
                        # file value can be referenced without being invoked.
                        target = declaration_for_read(node.id, scope)
                        if target and target['kind'] in ('parameter','local_variable','variable','field'):
                            pf.edge(owner, target, 'reads', node, 'resolved_declaration')
                        elif target is None:
                            target = resolved_name(node.id, scope, ())
                            if target:
                                pf.edge(owner, target, 'references', node, 'resolved_declaration')
        # Bases reference class declarations; subclass dispatch is dynamic.
        for cls in ast.walk(pf.tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            owner = pf.by_ast.get(id(cls))
            if not owner:
                continue
            for base in cls.bases:
                parts = _name_and_attributes(base)
                if parts:
                    target = resolved_name(parts[0], owner.get('parent', pf.file_node['id']), parts[1])
                    if target and target['kind'] == 'class':
                        pf.edge(owner, target, 'inherits', base, 'resolved_static')


def _js_declarations(source: str, path: str) -> list[dict]:
    """Lexically supported JS/TS declarations; excludes comments and strings.

    A lexer is not a compiler. Never produce exact `calls` from these tokens.
    """
    from analyzer.plugins.javascript_syntax import tokens
    ts = tokens(source)
    found = []
    kinds = {"class": "class", "function": "function", "interface": "interface",
             "type": "type", "enum": "enum"}
    for i, tok in enumerate(ts[:-1]):
        if tok.kind != "word" or tok.value not in kinds:
            continue
        next_tok = ts[i+1]
        if next_tok.kind != "word":
            continue
        if tok.value == "type" and (i+2 >= len(ts) or ts[i+2].value != "="):
            continue
        if tok.value == "function" and (i+2 >= len(ts) or ts[i+2].value not in ("(", "<")):
            continue
        # Filter simple property usages (`obj.class`) and imported named bindings.
        if i and ts[i-1].value in (".", "?.", "import"):
            continue
        line = source.count("\n", 0, tok.start) + 1
        found.append({"name": next_tok.value, "kind": kinds[tok.value], "line": line,
                      "confidence": "lexical_declaration"})
    return found


def _powershell_file_graphs(root: Path, candidates: dict, graph_data: dict) -> tuple[dict, list]:
    """Syntax-aware PowerShell symbol graphs; uncertain dispatch stays a candidate.

    Static cross-file resolution requires an existing file dependency edge.
    We never link unrelated functions only because their names match.
    """
    from analyzer.powershell_semantics import analyze_path
    grouped = {}
    warnings = []
    ps_files = {path for path, info in candidates.items() if info.get("language") == "powershell"}
    for path in sorted(ps_files):
        source = root / path
        if not source.is_file() or source.stat().st_size > 2_000_000:
            warnings.append({"path": path, "reason": "missing_or_source_exceeds_2mb"})
            continue
        try:
            data = analyze_path(source)
            grouped[path] = data
            if data.get('ast_fallback_reason'):
                warnings.append({'path':path,'reason':data['ast_fallback_reason']})
            for error in data.get('errors', []):
                warnings.append({"path": path, "reason": "powershell_parse_error", "details": error})
        except (OSError, UnicodeError, ValueError) as e:
            warnings.append({"path": path, "reason": type(e).__name__, "details": str(e)[:160]})
    symbol_table = {}
    for path, data in grouped.items():
        table = defaultdict(list)
        for function in data.get('functions', []):
            name = str(function.get('name', ''))
            if name:
                line = int(function.get('line', 1))
                table[name.casefold()].append({"id": _node_id(path, name, line, 0),
                    "path": path, "name": name, "qualname": name, "kind": "function", "line": line,
                    "end_line": int(function.get('end_line') or line), "parent": _file_id(path),
                    "confidence": "parsed_ast" if data['parser'] == 'powershell_ast' else "lexical_declaration"})
        symbol_table[path] = table
    imports = defaultdict(set)
    for edge in graph_data.get('edges', []):
        if edge.get('type') == 'script_source' and edge.get('source') in ps_files and edge.get('target') in ps_files:
            imports[edge['source']].add(edge['target'])
    results = {}
    for path, data in grouped.items():
        original = (root / path).read_bytes()
        lines = original.decode('utf-8-sig', errors='replace').splitlines()
        file_hash = _sha(original)
        nodes = [{"id": _file_id(path), "path": path, "name": path.rsplit('/',1)[-1],
                  "kind": "file", "line": 1, "end_line": len(lines), "confidence": "verified_file"}]
        edges = []
        known = set()
        by_id = {}
        def add_node(n):
            if n['id'] not in by_id:
                nodes.append(n)
                by_id[n['id']] = n
        def add_edge(origin, target, relation, line, resolution):
            if (origin['id'],target['id'],relation,line) in known:
                return
            known.add((origin['id'],target['id'],relation,line))
            snippet = lines[line-1].strip()[:200] if 1 <= line <= len(lines) else ''
            edges.append({"source": origin['id'], "target": target['id'],
                "source_path": origin['path'], "target_path": target['path'],
                "relation": relation, "resolution": resolution,
                "analyzer": data['parser'],
                "evidence": {"path": path, "line": line, "source_sha256": file_hash, "snippet": snippet}})
        file_node = nodes[0]
        for group in symbol_table[path].values():
            for sym in group:
                add_node(sym)
                add_edge(file_node, sym, 'declares', sym['line'], sym['confidence'])
        for fn in data.get('functions', []):
            for param in fn.get('parameters', []):
                name = str(param)
                owners = symbol_table[path].get(fn.get('name', '').casefold(), [])
                owner = next((x for x in owners if x['line']==fn['line']),None)
                if not owner or not name:
                    continue
                local = {"id": _node_id(path, owner['qualname'] + '.param:' + name, owner['line'], 0),
                    "path": path, "name": name, "qualname": owner['qualname']+'.param:'+name,
                    "kind": "parameter", "line": owner['line'], "parent": owner['id'],
                    "confidence": owner['confidence']}
                add_node(local)
                add_edge(owner, local, 'has_parameter', owner['line'], owner['confidence'])
        # Assignment edges from AST, or lexical candidates as documented.
        for assignment in data.get('assignments', []):
            owner_name = (assignment.get('owner') or '').casefold()
            owners = symbol_table[path].get(owner_name, [])
            owner = owners[0] if len(owners)==1 else file_node
            name = str(assignment.get('name',''))
            line = int(assignment.get('line',1))
            if not name:continue
            local = {"id": _node_id(path,owner['id']+'.variable:'+name,line,0),
                "path":path,"name":name,"qualname":owner['name']+'.variable:'+name,
                "kind":"local_variable","line":line,"parent":owner['id'],
                "confidence":"parsed_ast" if data['parser']=='powershell_ast' else 'lexical_candidate'}
            add_node(local)
            add_edge(owner,local,'writes',line,'parsed_ast' if data['parser']=='powershell_ast' else 'lexical_candidate')
        for command in data.get('commands', []):
            name = str(command.get('name','')).casefold()
            line = int(command.get('line',1))
            if not name:continue
            owner_name = str(command.get('owner') or '').casefold()
            owners = symbol_table[path].get(owner_name, [])
            owner = owners[0] if len(owners)==1 else file_node
            targets = symbol_table[path].get(name, [])
            if len(targets)==1:
                add_edge(owner,targets[0],'calls',line,
                         'ast_candidate_dispatch' if data['parser']=='powershell_ast' else 'lexical_candidate')
                continue
            if targets:continue
            # Explicitly imported or dot-sourced files ONLY. If more than one
            # definition matches, leave it unresolved instead of guessing.
            external = [sym for other in imports[path] for sym in symbol_table.get(other,{}).get(name,[])]
            if len(external)==1:
                symbol = {**external[0],"external":True}
                add_node(symbol)
                add_edge(owner,symbol,'call_candidate',line,
                         'ast_candidate_imported' if data['parser']=='powershell_ast' else 'lexical_candidate_imported')
        reason=data.get('ast_fallback_reason')
        results[path] = {"nodes":nodes,"edges":edges,"parser":data['parser'], "hash":file_hash,
            "ast_fallback_reason":reason,
            "caution": "PowerShell AST and statically located commands. Runtime imports and dispatch are not proven." if data['parser']=='powershell_ast'
             else "PowerShell lexical fallback (AST transport failed): all calls are candidates, not verified semantic edges." if reason
             else "PowerShell lexical fallback (no pwsh/powershell available): function calls and variable writes are candidates, not verified semantic edges."}
    return results, warnings


def build_symbol_graphs(root: Path, index: dict, graph_data: dict, output: Path) -> dict:
    """Build per-file graphs and an index; no source truncation is silent."""
    from analyzer.inventory import determine_language
    output.mkdir(parents=True, exist_ok=True)
    candidates = {item["path"]: item for item in index.get("files", [])}
    # Project-neutral adapters: one contract for Python, TypeScript, PowerShell,
    # Java, C and C++; no assumptions about repository or module names.
    from analyzer.semantic_adapters import AdapterContext, run_adapters
    sem_graphs, errors, adapter_report = run_adapters(AdapterContext(root, candidates, graph_data))
    # Rejected syntax must NEVER be replaced with legacy name-guessing. The
    # inventory may recognize a file but cannot establish its declarations.
    # Keep source-file navigation only and expose the reason in both exports.
    rejection_reasons = {
        'unparsed_source', 'xml_source_rejected', 'SyntaxError',
        'clojure_reader_unavailable_or_invalid_subset', 'php_parse_error',
        'source_exceeds_limit', 'adapter_failed', 'python_ast_file_failed',
        'UnicodeDecodeError', 'UnicodeError', 'RecursionError',
        'OverflowError', 'ValueError', 'OSError',
        'unsafe_source_path', 'source_exceeds_inventory_limit',
        'typescript_ast_parse_errors', 'go_parse_error', 'ruby_parse_error',
        'kotlin_psi_source_rejected',
    }
    rejected_by_adapter = {
        error['path']: error.get('reason', 'adapter_error')
        for error in errors
        if (error.get('path') in candidates and
            error.get('reason') in rejection_reasons and
            error.get('path') not in sem_graphs)
    }
    # Preserve the adapter's strict contract: invalid Clojure source is NOT
    # returned as a semantic graph. Read-only physical recovery belongs only to
    # the original Inside File *file-only* export, after reader rejection.
    rejected_clojure = {e.get('path') for e in errors
                        if e.get('reason') == 'clojure_reader_unavailable_or_invalid_subset'
                        and e.get('path') in candidates
                        and candidates[e['path']].get('language') == 'clojure'
                        and e.get('details') not in ('missing_symlink_or_outside_root',
                                                      'source_exceeds_limit')}
    recovered_csts = {}
    if rejected_clojure:
        from analyzer.semantic_adapters.clojure_cst_recovery import (
            recover_rejected_source_cst, MAX_RECOVERY_BYTES)
        for rejected_path in sorted(rejected_clojure):
            physical = root / rejected_path
            if physical.is_symlink() or not physical.is_file() or not physical.resolve().is_relative_to(root):
                continue
            try:
                if physical.stat().st_size > MAX_RECOVERY_BYTES:
                    continue
                raw = physical.read_bytes()
            except OSError:
                continue
            if len(raw) > MAX_RECOVERY_BYTES:
                continue
            try:
                raw.decode('utf-8-sig')
                category = 'structural_reader_rejected'
            except UnicodeDecodeError:
                category = 'invalid_utf8'
            platform = ('shared' if rejected_path.endswith('.cljc') else
                        'cljs' if rejected_path.endswith('.cljs') else 'clj')
            recovered_csts[rejected_path] = recover_rejected_source_cst(
                raw, platform=platform, reader_error_category=category)
    # Keep cross-file targets navigable in the inside-file graph. These are
    # references to parsed definitions, not guessed names.
    known_symbols = {node['id']: node for g in sem_graphs.values() for node in g['nodes']}
    for graph in sem_graphs.values():
        local_ids = {n['id'] for n in graph['nodes']}
        for edge in graph['edges']:
            for symbol_id in (edge['source'], edge['target']):
                if symbol_id not in local_ids and symbol_id in known_symbols:
                    graph['nodes'].append({**known_symbols[symbol_id], 'external': True})
                    local_ids.add(symbol_id)
    related: dict[str, list[dict]] = defaultdict(list)
    for edge in graph_data.get("edges", []):
        related[edge["source"]].append(edge)
        if edge["source"] != edge["target"]:
            related[edge["target"]].append(edge)
    # Promote only explicitly resolved symbol-to-symbol cross-file relationships.
    # This is generated from independent language semantics, never name similarity.
    promoted = []
    candidate_links = []
    for origin in sem_graphs.values():
        for e in origin["edges"]:
            if (e.get("source_path") != e.get("target_path") and
                    e.get("relation") in {"call_candidate", "jni_name_match", "jni_register_natives", "include_candidate", "sql_table_candidate", "clojure_require_candidate", "clojure_call_candidate", "clojure_macro_invocation_candidate", "clojure_macro_phase_candidate", "clojure_referred_call_candidate", "clojure_referred_macro_invocation_candidate", "clojure_referred_macro_phase_candidate", "module_candidate"} and
                    e.get("resolution") in {"psi_name_candidate", "candidate_jni_abi_name", "candidate_jni_abi_signature", "go_ast_name_candidate", "ruby_require_candidate", "candidate_dispatch", "candidate_jni_registration", "php_literal_include_candidate", "sql_unique_decl_candidate", "clojure_literal_ns_candidate", "clojure_qualified_name_candidate", "clojure_literal_refer_scope_candidate", "candidate_import_path"}):
                # Separate optional candidate graph layer. It is NOT static
                # binding proof, even if the names match exactly.
                candidate_links.append({"source":e["source_path"],"target":e["target_path"],
                    "relation":e["relation"],"analyzer":e["analyzer"],"evidence":e["evidence"]})
            if (e.get("source_path") == e.get("target_path") or
                    e.get("resolution") != "resolved_static" or
                    e.get("relation") not in {"calls", "inherits", "instantiates"}):
                continue
            promoted.append({"source":e["source_path"],"target":e["target_path"],
                             "relation":e["relation"], "analyzer":e["analyzer"],"evidence":e["evidence"]})
            edge = {"source":e["source_path"],"target":e["target_path"],
                    "type":"symbol_"+e["relation"],"confidence_v3":"resolved_static",
                    "analyzer":e["analyzer"],"evidence":[e["evidence"]]}
            related[edge["source"]].append(edge)
            related[edge["target"]].append(edge)
    # Candidate edges are deliberately separate from confirmed global graph
    # relationships. They must nevertheless be visible in the Inside File
    # "related files" inspector, with the same source-line evidence.
    # Previously only resolved/legacy graph relationships were copied into
    # file_edges, making a genuine *candidate* appear as "no related files".
    for e in candidate_links:
        rel={"source":e["source"],"target":e["target"],
             "type":e["relation"],"confidence_v3":"candidate",
             "analyzer":e["analyzer"],"evidence":[e["evidence"]]}
        related[rel["source"]].append(rel)
        if rel["target"] != rel["source"]:
            related[rel["target"]].append(rel)
    catalogue = {"schema": SCHEMA, "project": root.name, "source_commit": graph_data.get("source_commit", ""),
                 "source_mode": graph_data.get("metadata", {}).get("source_mode", "local"),
                 "files": {}, "errors": errors, "coverage": {}}
    coverage = Counter()
    original_nodes = {n.get("id"): n for n in graph_data.get("nodes", [])}
    for path, entry in sorted(candidates.items()):
        file = (root / path).resolve()
        if path in sem_graphs:
            result = sem_graphs[path]
            nodes, edges = result['nodes'], result['edges']
            parser, file_hash = result['parser'], result['hash']
            coverage[parser] += 1
            caution = result.get('caution','Source-evidenced static analysis; not runtime proof.')
        else:
            # Declared symbols from the pre-existing language plugins are NOT
            # semantic resolutions. Mark them as provisional declarations.
            raw = original_nodes.get(path, {})
            # An invalid path must not be read again by the legacy fallback,
            # even when a symlink happens to point *inside* the root.
            unsafe_path = rejected_by_adapter.get(path) in (
                'unsafe_source_path', 'source_exceeds_inventory_limit')
            file_hash = (_sha(file.read_bytes()) if not unsafe_path and
                         file.is_file() and file.is_relative_to(root) and
                         file.stat().st_size <= 2_000_000 else "")
            nodes = [{"id": _file_id(path), "path": path, "name": path.split("/")[-1],
                      "kind": "file", "line": 1, "confidence": "verified_file"}]
            edges = []
            if entry.get("language") in ("javascript", "typescript") and file_hash:
                declarations = _js_declarations(file.read_text(encoding="utf-8-sig", errors="replace"), path)
            else:
                declarations = raw.get("symbols", [])
            if path in recovered_csts or path in rejected_by_adapter:
                declarations = []  # No guessed symbols for rejected syntax.
            for s in declarations:
                if not isinstance(s, dict) or not s.get("name"):
                    continue
                line = int(s.get("line", 1) or 1)
                sid = _node_id(path, str(s["name"]), line, 0)
                nodes.append({"id": sid, "path": path, "name": s["name"],
                              "qualname": s.get("fqn", s["name"]), "kind": s.get("kind", "declaration"),
                              "line": line, "parent": _file_id(path), "confidence": s.get("confidence", "heuristic_declaration")})
                edges.append({"source": _file_id(path), "target": sid,
                              "source_path": path, "target_path": path,
                              "relation": "declares", "resolution": s.get("confidence", "heuristic_declaration"), "analyzer": "javascript-lexer" if entry.get("language") in ("javascript", "typescript") else "legacy-symbol-index",
                              "evidence": {"path": path, "line": line, "source_sha256": file_hash}})
            parser = "javascript_lexical" if entry.get("language") in ("javascript", "typescript") else ("declarations_only" if len(nodes) > 1 else "file_only")
            if path in recovered_csts:
                parser = 'clojure_rejected_source_physical_cst_only'
            elif path in rejected_by_adapter:
                parser = 'file_only'
            coverage[parser] += 1
            caution = ("Reader-rejected Clojure source: file-only physical CST, no semantic symbols/edges."
                       if path in recovered_csts else
                       "Declarations are heuristic or unavailable; no semantic call graph claimed.")
        # Existing file edges are shown as file-level references, not made up as calls.
        file_edges = []
        for e in related.get(path, []):
            file_edges.append({"source": _file_id(e["source"]), "target": _file_id(e["target"]),
                               "source_path": e["source"], "target_path": e["target"],
                               "relation": "file_" + str(e["type"]), "resolution": e.get("confidence_v3", "candidate"),
                               "analyzer": e.get("analyzer", "legacy"), "evidence": (e.get("evidence") or [{}])[0]})
        if path in recovered_csts or path in rejected_by_adapter:
            # No file relationships inferred from an invalid source, including
            # the legacy graph generated before semantic parser rejection.
            file_edges = []
        fingerprint = _sha(path.encode("utf-8"))[:24]
        file_path = "symbol-graphs/" + fingerprint + ".json"
        source_parse_status = ('parsed_partial' if path in sem_graphs else
                               'rejected_or_unavailable' if path in rejected_by_adapter else
                               'heuristic_or_unavailable')
        payload = {"schema": SCHEMA, "file": path, "language": entry.get("language", determine_language(path)),
                   "source_parse_status": source_parse_status,
                   "source_parse_issue": rejected_by_adapter.get(path),
                   "parser": parser, "parser_version": (sem_graphs.get(path) or {}).get('parser_version'),
                   "source_sha256": file_hash, "nodes": nodes,
                   "python_source_syntax": (sem_graphs.get(path) or {}).get('python_source_syntax', {'status':'not_applicable'}),
                   "edges": edges, "file_edges": file_edges,
                   "caution": caution,
                   "ast_fallback_reason": (sem_graphs.get(path) or {}).get("ast_fallback_reason"),
                   "semantic_overlay": (sem_graphs.get(path) or {}).get("semantic_overlay", ""),
                   "reader_limitations": (sem_graphs.get(path) or {}).get("reader_limitations", []),
                   "reader_grammar_evidence": (sem_graphs.get(path) or {}).get("reader_grammar_evidence", {"status":"not_applicable"}),
                   "clojure_source_cst": recovered_csts.get(path,
                       (sem_graphs.get(path) or {}).get("clojure_source_cst", {"status":"not_applicable"})),
                   "external_compiler_report_evidence": (sem_graphs.get(path) or {}).get("external_compiler_report_evidence", {"status":"not_requested"}),
                   "project_refer_all_evidence": (sem_graphs.get(path) or {}).get("project_refer_all_evidence", []),
                   "official_reader_evidence": (sem_graphs.get(path) or {}).get("official_reader_evidence", {"status":"not_requested"}),
                   "core_var_oracle_evidence": (sem_graphs.get(path) or {}).get("core_var_oracle_evidence", {"status":"not_requested"}),
                   "clojure_semantic_analysis": (sem_graphs.get(path) or {}).get("clojure_semantic_analysis", {}),
                   "clojure_source_form_ast": (sem_graphs.get(path) or {}).get("clojure_source_form_ast", {}),
                   "cljc_projected_form_asts": (sem_graphs.get(path) or {}).get("cljc_projected_form_asts", {}),
                   "clojure_project_type_hint_candidates": (sem_graphs.get(path) or {}).get("clojure_project_type_hint_candidates", {}),
                   "clojure_interprocedural_source_cycle_candidates": (sem_graphs.get(path) or {}).get("clojure_interprocedural_source_cycle_candidates", {}),
                   "clojure_interprocedural_argument_candidates": (sem_graphs.get(path) or {}).get(
                       "clojure_interprocedural_argument_candidates", {"status":"not_applicable", "calls":[]}),
                   "jvm_bytecode_interop_evidence": (sem_graphs.get(path) or {}).get("jvm_bytecode_interop_evidence", {"status":"not_requested", "sites":[]}),
                   "cljs_js_interop_evidence": (sem_graphs.get(path) or {}).get("cljs_js_interop_evidence", {"status":"not_requested", "sites":[]}),
                   "cljs_js_module_evidence": (sem_graphs.get(path) or {}).get("cljs_js_module_evidence", {"status":"not_requested", "sites":[]}),
                   "cljc_jvm_interop_evidence": (sem_graphs.get(path) or {}).get("cljc_jvm_interop_evidence", {"status":"not_applicable"}),
                   "cljc_js_interop_evidence": (sem_graphs.get(path) or {}).get("cljc_js_interop_evidence", {"status":"not_applicable"}),
                   "cljc_js_module_evidence": (sem_graphs.get(path) or {}).get("cljc_js_module_evidence", {"status":"not_applicable"}),
                   "declaration_limitations": (sem_graphs.get(path) or {}).get("declaration_limitations", []),
                   "source_coordinates": (sem_graphs.get(path) or {}).get("source_coordinates", "unspecified"),
                   # Namespace-bound XML source identifiers are structural
                   # evidence, not general IDREF/schema/runtime resolution.
                   "xml_id_index": (sem_graphs.get(path) or {}).get("xml_id_index", {}),
                   "xml_id_issues": (sem_graphs.get(path) or {}).get("xml_id_issues", []),
                   "unresolved_references": (sem_graphs.get(path) or {}).get("unresolved_references", []),
                   "unresolved_reference_count": (sem_graphs.get(path) or {}).get("unresolved_reference_count", 0),
                   # Stage42.1: explicitly expose aliases that cannot be used
                   # as proven runtime namespace bindings.
                   "namespace_alias_guard_reasons": (sem_graphs.get(path) or {}).get("namespace_alias_guard_reasons", {}),
                   "private_var_guard": (sem_graphs.get(path) or {}).get("private_var_guard", "not_applicable"),
                   "platform_scope": (sem_graphs.get(path) or {}).get("platform_scope", "not_applicable"),
                   # Stage40/41 evidence must survive serialization, not be
                   # lost between the adapter and the Inside File JSON export.
                   # These ARE NOT graph edges or verified classpath loads.
                   "platform_projections": (sem_graphs.get(path) or {}).get("platform_projections", {}),
                   "clojure_project_context": (sem_graphs.get(path) or {}).get("clojure_project_context", {}),
                   "platform_namespace_evidence": (sem_graphs.get(path) or {}).get("platform_namespace_evidence", {}),
                   "platform_macro_phase_evidence": (sem_graphs.get(path) or {}).get("platform_macro_phase_evidence", {}),
                   "lexical_scope_limitations": (sem_graphs.get(path) or {}).get("lexical_scope_limitations", []),
                   "lexical_scope_suppression_count": (sem_graphs.get(path) or {}).get("lexical_scope_suppression_count", 0),
                   "qualified_scope_suppression_count": (sem_graphs.get(path) or {}).get("qualified_scope_suppression_count", 0),
                   "qualified_scope_suppression_examples": (sem_graphs.get(path) or {}).get("qualified_scope_suppression_examples", []),
                   "lexical_scope_suppression_examples": (sem_graphs.get(path) or {}).get("lexical_scope_suppression_examples", []),
                   "semantic_completeness": "not_certified" }
        dest = output / file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        catalogue["files"][path] = {"graph": file_path, "parser": parser,
                                     "source_parse_status": source_parse_status,
                                     "source_parse_issue": rejected_by_adapter.get(path),
                                     "symbol_count": sum(n.get("path") == path and n.get("kind") not in {"file", "clojure_syntax_form",
                                        "clojure_call_result_candidate", "clojure_formal_pattern_candidate",
                                        "clojure_actual_argument_candidate",
                                        "clojure_possible_return_expression"} for n in nodes),
                                     "syntax_form_count": sum(n.get("path") == path and n.get("kind") == "clojure_syntax_form" for n in nodes),
                                     "internal_edges": len(edges), "file_links": len(file_edges), "source_sha256": file_hash,
                                     "semantic_overlay": (sem_graphs.get(path) or {}).get("semantic_overlay", ""),
                                     "ast_fallback_reason": (sem_graphs.get(path) or {}).get("ast_fallback_reason"),
                                     "reader_limitations": (sem_graphs.get(path) or {}).get("reader_limitations", []),
                                     "clojure_source_cst_status": recovered_csts.get(path,
                                         (sem_graphs.get(path) or {}).get("clojure_source_cst", {})).get("status", "not_applicable"),
                                     "external_compiler_report_status": (sem_graphs.get(path) or {}).get("external_compiler_report_evidence", {}).get("status", "not_requested"),
                                     "project_refer_all_evidence": (sem_graphs.get(path) or {}).get("project_refer_all_evidence", []),
                                     "clojure_source_cst_node_count": recovered_csts.get(path,
                                         (sem_graphs.get(path) or {}).get("clojure_source_cst", {})).get("node_count", 0),
                   "official_reader_evidence": (sem_graphs.get(path) or {}).get("official_reader_evidence", {"status":"not_requested"}),
                   "core_var_oracle_evidence": (sem_graphs.get(path) or {}).get("core_var_oracle_evidence", {"status":"not_requested"}),
                   "clojure_semantic_analysis": (sem_graphs.get(path) or {}).get("clojure_semantic_analysis", {}),
                   "clojure_source_form_ast_status": (sem_graphs.get(path) or {}).get("clojure_source_form_ast", {}).get("status", "not_applicable"),
                   "clojure_source_form_ast_node_count": (sem_graphs.get(path) or {}).get("clojure_source_form_ast", {}).get("node_count", 0),
                   "clojure_project_type_hint_candidate_count": len((sem_graphs.get(path) or {}).get("clojure_project_type_hint_candidates", {}).get("hints", [])),
                   "clojure_interprocedural_cycle_count": len((sem_graphs.get(path) or {}).get("clojure_interprocedural_source_cycle_candidates", {}).get("components", [])),
                   "clojure_interprocedural_argument_candidates": (sem_graphs.get(path) or {}).get(
                       "clojure_interprocedural_argument_candidates", {"status":"not_applicable", "calls":[]}),
                   "jvm_bytecode_interop_evidence": (sem_graphs.get(path) or {}).get("jvm_bytecode_interop_evidence", {"status":"not_requested", "sites":[]}),
                   "cljs_js_interop_evidence": (sem_graphs.get(path) or {}).get("cljs_js_interop_evidence", {"status":"not_requested", "sites":[]}),
                   "cljs_js_module_evidence": (sem_graphs.get(path) or {}).get("cljs_js_module_evidence", {"status":"not_requested", "sites":[]}),
                   "cljc_jvm_interop_evidence": (sem_graphs.get(path) or {}).get("cljc_jvm_interop_evidence", {"status":"not_applicable"}),
                   "cljc_js_interop_evidence": (sem_graphs.get(path) or {}).get("cljc_js_interop_evidence", {"status":"not_applicable"}),
                   "cljc_js_module_evidence": (sem_graphs.get(path) or {}).get("cljc_js_module_evidence", {"status":"not_applicable"}),
                                     "declaration_limitations": (sem_graphs.get(path) or {}).get("declaration_limitations", []),
                                     "source_coordinates": (sem_graphs.get(path) or {}).get("source_coordinates", "unspecified"),
                                     "xml_id_count": len((sem_graphs.get(path) or {}).get("xml_id_index", {})),
                                     "xml_id_issue_count": len((sem_graphs.get(path) or {}).get("xml_id_issues", [])),
                                     "unresolved_references": (sem_graphs.get(path) or {}).get("unresolved_references", []),
                                     "unresolved_reference_count": (sem_graphs.get(path) or {}).get("unresolved_reference_count", 0),
                                     "platform_scope": (sem_graphs.get(path) or {}).get("platform_scope", "not_applicable"),
                                     "macro_phase_candidate_count": (sem_graphs.get(path) or {}).get("platform_macro_phase_evidence", {}).get("candidate_count",0),
                                     "platform_candidate_counts": {
                                         platform: details.get("candidate_count", 0)
                                         for platform, details in (sem_graphs.get(path) or {}).get("platform_namespace_evidence", {}).items()
                                     },
                                     "lexical_scope_limitations": (sem_graphs.get(path) or {}).get("lexical_scope_limitations", []),
                                     "lexical_scope_suppression_count": (sem_graphs.get(path) or {}).get("lexical_scope_suppression_count", 0),
                                     "qualified_scope_suppression_count": (sem_graphs.get(path) or {}).get("qualified_scope_suppression_count", 0),
                                     "qualified_scope_suppression_examples": (sem_graphs.get(path) or {}).get("qualified_scope_suppression_examples", []),
                                     "semantic_completeness": "not_certified" }
    catalogue["coverage"] = dict(coverage)
    catalogue["source_parse_status_counts"] = dict(Counter(
        entry["source_parse_status"] for entry in catalogue["files"].values()))
    catalogue["semantic_adapters"] = adapter_report
    catalogue["promoted_symbol_reference_count"] = len(promoted)
    # Integrity is structural and evidentiary, not proof of semantic completeness.
    invalid = []
    checked = 0
    for source_path, info in catalogue["files"].items():
        document = json.loads((output / info["graph"]).read_text(encoding="utf8"))
        ids = {node["id"] for node in document["nodes"]}
        for edge in document["edges"]:
            checked += 1
            if edge["source"] not in ids or edge["target"] not in ids:
                invalid.append({"file": source_path, "problem": "missing_symbol_endpoint", "relation": edge["relation"]})
            ev = edge.get("evidence", {})
            if ev.get("path") != source_path or ev.get("source_sha256") != document["source_sha256"]:
                invalid.append({"file": source_path, "problem": "bad_symbol_provenance", "relation": edge["relation"]})
            max_line = int(candidates[source_path].get("lines", 0))
            if (not isinstance(ev.get("line"), int) or ev["line"] < 1
                    or (max_line and ev["line"] > max_line)):
                invalid.append({"file": source_path, "problem": "invalid_line", "relation": edge["relation"]})
    integrity = {"schema": SCHEMA, "valid": not invalid, "checked_internal_edges": checked,
                 "problems": invalid[:200],
                 "scope": "Endpoint and provenance checks only; runtime behavior and dynamic dispatch are NOT validated."}
    (output / "SYMBOL_INTEGRITY.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (output / "symbol-index.json").write_text(json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalogue["_promoted_edges"] = promoted
    catalogue["_candidate_symbol_edges"] = candidate_links
    return catalogue
