"""PowerShell dependency analysis, separating literal imports from dynamic calls.

Literal Import-Module / dot-source paths can be resolved. A wildcard reference
is only a *candidate*. Cross-file calls require an explicit import relation and
unique declaration; PowerShell's dynamic scope prevents claiming runtime truth.
"""
from __future__ import annotations
import fnmatch
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from . import Context
from analyzer.model import EdgeType, ConfidenceLevel
from analyzer.powershell_semantics import analyze_path, strip_comments_strings
from analyzer.powershell_dataflow import detect_dependencies

SOURCE = re.compile(r'^\s*(?:\.\s+|&\s+|Import-Module\s+(?:-Name\s+)?|using\s+module\s+)(.+)',re.I)
MANIFEST = re.compile(r'^\s*(RootModule|ModuleToProcess|NestedModules|ScriptsToProcess|RequiredModules)\s*=\s*(.+)',re.I)
QUOTED = re.compile(r"['\"]([^'\"]+\.(?:ps1|psm1|psd1))['\"]", re.I)
MODULE_NAME = re.compile(r'^([\w.-]+)(?:\s|$)',re.I)
FILENAME = re.compile(r'[\w.$/\\*{}-]+\.(?:ps1|psm1|psd1)',re.I)
JOIN_PATH = re.compile(r'Join-Path\s+(?:-Path\s+)?\$PSScriptRoot\s+(?:-ChildPath\s+)?[\"\']([^\"\']+)[\"\']',re.I)
# A wildcard enumerated and passed to Import-Module can establish candidates,
# but never proof that the loop actually executes.
DYNAMIC_IMPORT = re.compile(r'\bImport-Module\b',re.I)
GET_CHILD = re.compile(r'\b(?:Get-ChildItem|gci|dir|ls)\b',re.I)


def candidate_paths(ref: str,directory: str,paths: set[str]) -> list[str]:
    ref=ref.replace('\\','/').replace('${PSScriptRoot}','$PSScriptRoot')
    ref=re.sub(r'^\$PSScriptRoot\s*/?', '',ref,flags=re.I)
    ref=ref.removeprefix('./')
    if ':' in ref or ref.startswith('/') or '$' in ref:return []
    target=posixpath.normpath(posixpath.join(directory,ref))
    if target.startswith('../'):return []
    lookup={p.lower():p for p in paths}
    if '*' in target or '?' in target:
        return sorted(p for p in paths if fnmatch.fnmatch(p.lower(),target.lower()))
    hit=lookup.get(target.lower())
    return [hit] if hit else []


def _without_inline_comment(line: str) -> str:
    quote = None
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == '`':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '\"'):
            quote = char
        elif char == '#':
            return line[:i]
    return line


def _literal_refs(text: str, paths: set[str], source: str) -> list[tuple[str,int,str,ConfidenceLevel]]:
    lines=text.splitlines()
    clean=strip_comments_strings(text)
    directory=posixpath.dirname(source)
    known_ps=[p for p in paths if p.lower().endswith(('.psm1','.psd1'))]
    results=[]
    for no,(original,masked) in enumerate(zip(lines,clean),1):
        if not masked.strip():continue
        # Only actual command outside a comment/string, not a mention in a
        # comment. Source line holds the literal quoted argument.
        m=SOURCE.match(masked)
        manifest=MANIFEST.match(masked) if source.lower().endswith('.psd1') else None
        if not m and not manifest:continue
        original_match = SOURCE.match(_without_inline_comment(original)) if m else MANIFEST.match(_without_inline_comment(original))
        if not original_match: continue
        expr = original_match.group(1 if m else 2)
        # Remove trailing inline comments, but don't look beyond the expression.
        # For an ordinary quoted path this preserves the original characters.
        # PowerShell AST is used separately for command detection on supported OS.
        refs=[]
        j=JOIN_PATH.search(expr)
        if j:refs.append(j.group(1))
        refs += [x.group(1) for x in QUOTED.finditer(expr)]
        if not refs:
            bare=FILENAME.search(expr)
            if bare:refs.append(bare.group(0))
            elif m and re.search(r'\b(?:Import-Module|using\s+module)\b',masked,re.I):
                name=MODULE_NAME.search(expr.strip())
                if name:
                    matches=[p for p in known_ps if posixpath.splitext(posixpath.basename(p))[0].lower()==name.group(1).lower()]
                    if len(matches)==1:results.append((matches[0],no,original.strip(),ConfidenceLevel.CANDIDATE))
        for ref in set(refs):
            wildcard='*' in ref or '?' in ref
            for target in candidate_paths(ref,directory,paths):
                if target!=source:results.append((target,no,original.strip(),ConfidenceLevel.CANDIDATE if wildcard else ConfidenceLevel.RESOLVED_STATIC))
    # Repeated identical evidence is merged, distinct lines retained.
    return list(dict.fromkeys(results))


def _dynamic_candidates(text: str,paths:set[str],source:str) -> list[tuple[str,int,str,ConfidenceLevel]]:
    # Only if a file has BOTH a wildcard enumeration and a dynamic
    # Import-Module. Still only a candidacy: the runtime may filter files.
    lines=text.splitlines()
    masked=strip_comments_strings(text)
    if not any(DYNAMIC_IMPORT.search(x) for x in masked):return []
    directory=posixpath.dirname(source)
    found=[]
    for no,(raw,code) in enumerate(zip(lines,masked),1):
        if not GET_CHILD.search(code):continue
        for match in QUOTED.finditer(_without_inline_comment(raw)):
            ref=match.group(1)
            if not ('*' in ref or '?' in ref):continue
            for target in candidate_paths(ref,directory,paths):
                if target!=source:found.append((target,no,raw.strip(),ConfidenceLevel.CANDIDATE))
    return list(dict.fromkeys(found))


@dataclass
class PowerShellPlugin:
    name: str='powershell'
    languages:tuple[str,...]=('powershell',)

    def analyze(self,ctx:Context) -> None:
        files=sorted(ctx.files_by_language.get('powershell',[]))
        definitions={}
        references={}
        parsed={}
        for source in files:
            try:
                data=analyze_path(ctx.root/source)
                text=ctx.read(source)
            except OSError:continue
            parsed[source]=data
            if data.get('ast_fallback_reason'):
                ctx.miss(self.name,source,1,data['ast_fallback_reason'])
            definitions[source]={d['name'].lower() for d in data.get('functions',[]) if d.get('name')}
            refs=_literal_refs(text,ctx.paths,source)
            detected, unresolved = detect_dependencies(text,source,ctx.paths)
            refs += [(r.target,r.line,r.snippet,r.confidence) for r in detected]
            for miss in unresolved:ctx.miss(self.name,source,miss["line"],miss["reason"])
            # Deduplicate line-specific references, prefer static path resolution
            # over candidates; no filename or project-specific matching.
            unique={}
            for target,no,snippet,confidence in refs:
                key=(target,no)
                if key not in unique or confidence==ConfidenceLevel.RESOLVED_STATIC:unique[key]=(target,no,snippet,confidence)
            refs=list(unique.values());references[source]=refs
            for target,no,snippet,confidence in refs:
                ctx.add(source,target,no,snippet,self.name,EdgeType.SCRIPT_SOURCE,confidence)
        # A command name match alone never proves an external relationship.
        # Require a path-based or module-name import and unique matching target.
        for source,data in parsed.items():
            imported={t for t,_,_,_ in references.get(source,[])}
            by_name={}
            for target in imported:
                for name in definitions.get(target,[]):by_name.setdefault(name,[]).append(target)
            for command in data.get('commands',[]):
                key=str(command.get('name','')).lower()
                hits=set(by_name.get(key,[]))
                if len(hits)!=1:continue
                target=next(iter(hits))
                if target==source:continue
                line=int(command.get('line',1))
                content=ctx.read(source).splitlines()
                snippet=content[line-1].strip() if 0<line<=len(content) else key
                ctx.add(source,target,line,snippet,'powershell_symbol_calls',EdgeType.CALL_CANDIDATE,ConfidenceLevel.CANDIDATE)


PLUGIN=PowerShellPlugin()
