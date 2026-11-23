"""Conservative inter-statement abstract interpretation of PowerShell module loading.

No target project code is executed, no network access is used. This adapter is
language-semantic: names of commands/operators are PowerShell syntax, never
repo-specific filenames, functions, folders, or application identifiers.

It tracks small *sets of possible file paths*, not actual runtime values.
If values cannot be derived, it emits no edge. Enumerated files become only
candidate dependencies, because conditions, external state and runtime inputs
can change execution. Static paths are called statically resolved, not executed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import fnmatch
import posixpath
import re

from analyzer.powershell_semantics import strip_comments_strings
from analyzer.model import ConfidenceLevel

# Language keywords and built-in cmdlets, NOT names taken from any project.
_VAR = re.compile(r'\$(?:\{(?:(?:script|global|local|private):)?([\w-]+)\}|(?:(?:script|global|local|private):)?([\w-]+))(?:\.(\w+))?', re.I)
_ASSIGN = re.compile(r'^\s*\$(?:\{?(?:script|global|local|private):)?([\w-]+)\}?\s*=\s*(.*)$', re.I)
_JOIN = re.compile(r'\bJoin-Path\b', re.I)
_ENUM = re.compile(r'\b(?:Get-ChildItem|gci)\b', re.I)
_IMPORT = re.compile(r'\b(?:Import-Module|using\s+module)\b', re.I)
_FOREACH = re.compile(r'\bforeach\s*\(\s*\$(\w+)\s+in\s+\$(\w+)\s*\)', re.I)
_EACH = re.compile(r'\b(?:ForEach-Object|%)\s*\{',re.I)
_QUOTED = re.compile(r'''(?P<q>['"])(?P<val>(?:`.|(?! (?P=q) ).)*?)(?P=q)'''.replace(' ', ''), re.S)
_ARG = re.compile(r'''"(?:`.|[^"])*"|'(?:''|[^'])*'|\$\{?[-\w:]+\}?(?:\.\w+)?|[^\s|]+''')
_WHERE_CLAUSE = re.compile(r'''\$_\.(Name|FullName|Extension)\s+(-ne|-eq|-like|-notlike)\s+(['"])([^'"]+)\3''',re.I)


@dataclass(frozen=True)
class AbstractFiles:
    values: frozenset[str] = frozenset()
    enumerated: bool = False
    origins: tuple[int, ...] = ()


@dataclass(frozen=True)
class Dependency:
    target: str
    line: int
    snippet: str
    confidence: ConfidenceLevel
    origins: tuple[int, ...]


def _tokens(s: str) -> list[str]:
    return _ARG.findall(s)


def _unquote(s: str) -> str:
    return s[1:-1].replace("''", "'") if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"" else s


def _variable(s: str, env: dict[str,AbstractFiles]) -> AbstractFiles:
    m=_VAR.fullmatch(s.strip().strip('()'))
    if not m: return AbstractFiles()
    key=(m.group(1) or m.group(2) or '').casefold()
    val=env.get(key,AbstractFiles())
    if m.group(3) and m.group(3).casefold() not in ('fullname','name'):
        return AbstractFiles()
    return val


def _safe_norm(prefix: str, path: str) -> str | None:
    path=path.replace('\\','/')
    if '\x00' in path or ':' in path or path.startswith('/'):
        return None
    result=posixpath.normpath(posixpath.join(prefix,path))
    if result=='..' or result.startswith('../'):
        return None
    return result


def _literal_paths(token: str, env: dict[str,AbstractFiles], source_dir: str) -> AbstractFiles:
    token=token.strip().strip(',')
    if not token:
        return AbstractFiles()
    val=_variable(token,env)
    if val.values:return val
    # Parse expressions rooted in $PSScriptRoot, a defined PowerShell variable.
    # Tokens containing other unresolved variable substitutions are NOT treated
    # as known paths (even when there is coincidental filename similarity).
    raw=_unquote(token)
    m=re.match(r'^\$(?:\{)?PSScriptRoot(?:\})?([\\/].*)?$',raw,re.I)
    if m:
        candidate=_safe_norm(source_dir,(m.group(1) or '').lstrip('/\\'))
        return AbstractFiles(frozenset((candidate,)) if candidate else frozenset())
    if '$' in raw or '`' in raw:return AbstractFiles()
    candidate=_safe_norm(source_dir,raw)
    return AbstractFiles(frozenset((candidate,)) if candidate else frozenset())


def _get_flag(toks: list[str],name: str) -> str | None:
    wanted='-'+name.casefold()
    for i,t in enumerate(toks[:-1]):
        if t.casefold()==wanted:return toks[i+1]
    return None


def _eval_join(expr: str, env: dict[str,AbstractFiles], source_dir: str) -> AbstractFiles:
    m=_JOIN.search(expr)
    if not m:return AbstractFiles()
    toks=_tokens(expr[m.end():].strip().strip('()'))
    named_base=_get_flag(toks,'Path') or _get_flag(toks,'LiteralPath')
    named_child=_get_flag(toks,'ChildPath')
    positional=[t for t in toks if not t.startswith('-')]
    base_tok=named_base or (positional[0] if positional else None)
    child_tok=named_child or (positional[1] if len(positional)>1 else None)
    if not base_tok or not child_tok:return AbstractFiles()
    base=_literal_paths(base_tok,env,source_dir)
    child=_unquote(child_tok)
    if '$' in child or '`' in child:return AbstractFiles()
    vals=set()
    for b in base.values:
        norm=_safe_norm(b,child)
        if norm is not None:vals.add(norm)
    return AbstractFiles(frozenset(vals),base.enumerated,base.origins)


def _apply_filter(paths: set[str], expr: str) -> set[str]:
    # Only simple, fully recognized Where-Object predicates are applied.
    # Unknown predicates leave a *superset* of candidates instead of creating
    # a false assertion that excluded files cannot be loaded.
    for property_,op,_,needle in _WHERE_CLAUSE.findall(expr):
        property_=property_.lower()
        def value(path):
            return posixpath.basename(path) if property_=='name' else (posixpath.splitext(path)[1] if property_=='extension' else path)
        if op.lower() in ('-eq','-ne'):
            paths={p for p in paths if (value(p).casefold()==needle.casefold())==(op.lower()=='-eq')}
        else:
            paths={p for p in paths if fnmatch.fnmatch(value(p).casefold(),needle.casefold())==(op.lower()=='-like')}
    return paths


def _eval_enum(expr: str,env: dict[str,AbstractFiles],source_dir: str,known: set[str]) -> AbstractFiles:
    m=_ENUM.search(expr)
    if not m:return AbstractFiles()
    first=expr[m.end():].split('|',1)[0]
    toks=_tokens(first)
    flag_path=_get_flag(toks,'Path') or _get_flag(toks,'LiteralPath')
    positional=next((t for t in toks if not t.startswith('-')),None)
    path_expr=flag_path or positional
    if not path_expr:return AbstractFiles()
    # -Filter alone does not specify a search root. It still uses PS cwd,
    # which cannot safely be inferred from a static file path in general.
    base=_literal_paths(path_expr,env,source_dir)
    if not base.values:return AbstractFiles()
    filt=_get_flag(toks,'Filter')
    include=_get_flag(toks,'Include')
    exclude=_get_flag(toks,'Exclude')
    recurse='-recurse' in (t.lower() for t in toks)
    vals=set()
    for root in base.values:
        for path in known:
            if path == root:continue
            if '*' in root or '?' in root:
                if not fnmatch.fnmatch(path.casefold(),root.casefold()):continue
            else:
                if not path.startswith(root.rstrip('/')+'/'):continue
                rel=path[len(root.rstrip('/')+'/'):]
                if not recurse and '/' in rel:continue
            base_name=posixpath.basename(path)
            if filt and not fnmatch.fnmatch(base_name.casefold(),_unquote(filt).casefold()):continue
            if include and not fnmatch.fnmatch(base_name.casefold(),_unquote(include).casefold()):continue
            if exclude and fnmatch.fnmatch(base_name.casefold(),_unquote(exclude).casefold()):continue
            vals.add(path)
    if '|\u00a0' in expr:pass
    if re.search(r'\|\s*(?:Where-Object|\?)\s*\{',expr,re.I):
        vals=_apply_filter(vals,expr)
    return AbstractFiles(frozenset(vals),True,base.origins)


def _expr(expr: str, env: dict[str,AbstractFiles], dirname: str, known: set[str]) -> AbstractFiles:
    if _JOIN.search(expr):return _eval_join(expr,env,dirname)
    if _ENUM.search(expr):return _eval_enum(expr,env,dirname,known)
    return _literal_paths(expr.strip(),env,dirname)


def detect_dependencies(text: str, source: str, known: set[str]) -> tuple[list[Dependency],list[dict]]:
    """Return *possible* intra-repository module imports with source evidence.

    The evaluator understands a small PowerShell subset. It does not execute
    the analyzed file or assume its control-flow branches will run.
    """
    dirname=posixpath.dirname(source)
    clean=strip_comments_strings(text)
    original=text.splitlines()
    env:dict[str,AbstractFiles]={"psscriptroot":AbstractFiles(frozenset({dirname or '.'}))}
    references=[];unresolved=[]
    pipeline: AbstractFiles | None=None
    for line_no,(raw,masked) in enumerate(zip(original,clean),1):
        if not masked.strip():continue
        # Do not treat quoted command names as actual commands.
        # PowerShell multiline comments are masked by strip_comments_strings.
        assignment=_ASSIGN.match(masked)
        if assignment:
            actual=_ASSIGN.match(raw)
            if actual:
                variable=assignment.group(1).casefold()
                value=_expr(actual.group(2),env,dirname,known)
                env[variable]=AbstractFiles(value.values,value.enumerated,tuple(sorted(set(value.origins+(line_no,)))))
                pipeline=env[variable] if value.enumerated else None
        iterator=_FOREACH.search(masked)
        if iterator:
            element,collection=iterator.group(1).casefold(),iterator.group(2).casefold()
            val=env.get(collection,AbstractFiles())
            if val.values:
                env[element]=AbstractFiles(val.values,True,tuple(sorted(set(val.origins+(line_no,)))))
            else:
                env.pop(element,None)
        if _ENUM.search(masked) and not assignment:
            # Handles `Get-ChildItem ... | ForEach-Object { Import-Module $_.FullName }`.
            # No edges are emitted unless the Import-Module site is present.
            pipeline=_eval_enum(raw,env,dirname,known)
            if pipeline.values:
                pipeline=AbstractFiles(pipeline.values,True,(line_no,))
        if _EACH.search(masked) and pipeline and pipeline.values:
            env['_']=pipeline
        # `Import-Module` is a command even when inside foreach/try/if. A
        # masked quoted mention cannot cause a dependency.
        for imp in _IMPORT.finditer(masked):
            # Avoid bare e.g. `$foo.Import-Module()` method names.
            if imp.start() and masked[imp.start()-1] in '.$':continue
            args=raw[imp.end():]
            toks=_tokens(args)
            # Unknown imports remain unresolved, NEVER linked by coincidental
            # matching filenames or names of declared functions.
            value_tok=_get_flag(toks,'Name')
            if not value_tok:
                value_tok=next((x for x in toks if x and not x.startswith('-') and x not in ('{','}')),None)
            if not value_tok:
                unresolved.append({'line':line_no,'reason':'import_without_static_target'});continue
            value=_expr(value_tok,env,dirname,known)
            targets={p for p in value.values if p in known and p!=source and p.casefold().endswith(('.ps1','.psm1','.psd1'))}
            if not targets:
                unresolved.append({'line':line_no,'reason':'import_target_not_statically_identifiable'});continue
            for target in sorted(targets):
                references.append(Dependency(target,line_no,raw.strip()[:200],
                    ConfidenceLevel.CANDIDATE if value.enumerated else ConfidenceLevel.RESOLVED_STATIC,
                    value.origins))
    return list(dict.fromkeys(references)),unresolved
