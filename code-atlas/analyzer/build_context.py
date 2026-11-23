"""Static, evidence-oriented build-context bridge.

Does not execute Gradle, Maven, CMake, shell or analyzed project code.
It recognizes only literal declarations, preserving scope and uncertainties.
It is not a substitute for a resolved Gradle/Maven classpath or Android variant API.
"""
from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .model import EdgeEvidence, EdgeType, ConfidenceLevel

_Q = r'["\']'
_SETTINGS_INCLUDE = re.compile(r'\binclude\s*(?:\(\s*)?["\'](:[\w.:-]+)["\']')
_PROJECT_DEP = re.compile(r'\b(?:api|implementation|compileOnly|runtimeOnly|testImplementation|androidTestImplementation|debugImplementation|releaseImplementation)\s*(?:\(\s*)?project\s*\(\s*["\'](:[\w.:-]+)["\']')
_CMAKE_PATH = re.compile(r'\bpath\s*(?:=\s*file\s*\(|\s+file\s*\(|\(\s*|["\'])\s*["\']([^"\']+)["\']')
_LOADLIB = re.compile(r'\bSystem\s*\.\s*loadLibrary\s*\(\s*["\']([A-Za-z0-9_.-]+)["\']\s*\)')
_LIBRARY = re.compile(r'^\s*add_library\s*\(\s*([A-Za-z][\w.-]*)\s+(SHARED|STATIC|MODULE)\b',re.I)
_VARIANT = re.compile(r'\b(?:create\s*\(\s*["\']([\w.-]+)["\']\s*\)|(?:debug|release)\s*\{|abiFilters\s*(?:\+?=|\(|\s+))')


def _lines(text):
    """Conservative stripping: avoid comment-only lines and block comments."""
    block = False
    for number, original in enumerate(text.splitlines(), 1):
        line = original
        output = ''
        i = 0
        quote = None
        while i < len(line):
            c = line[i]
            if block:
                if line[i:i+2] == '*/': block = False; i += 2
                else: i += 1
                continue
            if quote:
                output += c
                if c == '\\' and i+1 < len(line): output += line[i+1]; i += 2; continue
                if c == quote: quote = None
                i += 1; continue
            if line[i:i+2] == '/*': block = True; i += 2; continue
            if line[i:i+2] == '//': break
            if c in ('"', "'"): quote = c
            output += c; i += 1
        yield number, output, original.strip()


def _not_inside_string(code: str, index: int) -> bool:
    """Prevent false loadLibrary occurrences inside ordinary source string literals."""
    quote = None
    escaped = False
    for char in code[:index]:
        if escaped:
            escaped = False
        elif quote and char == '\\':
            escaped = True
        elif quote == char:
            quote = None
        elif quote is None and char in ('"', "'"):
            quote = char
    return quote is None


def _mask_quoted(code: str) -> str:
    """Preserve positions/braces outside quoted text for a conservative DSL scan."""
    chars = list(code)
    quote = None
    escaped = False
    for i, char in enumerate(code):
        if quote:
            chars[i] = ' '
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
            chars[i] = ' '
    return ''.join(chars)


def _cmake_declaration_context(code_lines, index, column):
    """Require nesting inside externalNativeBuild { cmake { ... } }.

    This conservative brace scanner does not evaluate Gradle. Unknown DSL
    constructs simply stay unresolved rather than manufacturing a link.
    """
    stack = []
    token = re.compile(r'\b([A-Za-z_][\w]*)\s*\{|[{}]')
    for i, (_, line, _) in enumerate(code_lines[:index + 1]):
        masked = _mask_quoted(line[:column] if i == index else line)
        for match in token.finditer(masked):
            if match.group(1):
                stack.append(match.group(1))
            elif match.group() == '{':
                stack.append('')
            elif match.group() == '}' and stack:
                stack.pop()
    try:
        external_index = stack.index('externalNativeBuild')
        cmake_index = stack.index('cmake', external_index + 1)
        return cmake_index > external_index
    except ValueError:
        return False


def _maven_children(node, name):
    return [child for child in node if child.tag.rsplit('}',1)[-1] == name]


def _maven_text(node, name):
    children = _maven_children(node, name)
    return (children[0].text or '').strip() if children else ''


def _maven_modules(text):
    root = ET.fromstring(text)
    modules = _maven_children(root,'modules')
    return [((c.text or '').strip()) for mods in modules for c in _maven_children(mods,'module') if (c.text or '').strip()]


def analyze_build_context(root: Path, graph) -> dict:
    """Add only evidence-backed file-to-file build edges; report unproven context."""
    known = set(graph.nodes)
    data = {}
    for path in known:
        if path.endswith(('build.gradle.kts','build.gradle','settings.gradle','settings.gradle.kts','CMakeLists.txt','pom.xml')) or path.endswith(('.kt','.java','.cpp','.cc','.c','.m','.mm')):
            try:
                p = root / path
                if not p.is_file() or p.is_symlink() or p.stat().st_size > 2_000_000: continue
                data[path] = p.read_text(encoding='utf8', errors='replace')
            except (OSError, UnicodeError): pass
    summary = {'parser':'atlas_static_build_context_v2_scoped','execution':'none', 'links':[], 'variants':[], 'unresolved':[], 'note':'Declared build relations only; effective classpath and selected variants require Gradle/Maven model.'}
    def edge(src,dst,kind,confidence,ln,snip,reason):
        if src not in known or dst not in known: return
        item = graph.add_edge(src,dst,kind,confidence,'build_context',EdgeEvidence(src,ln,snippet=snip))
        if item: summary['links'].append({'from':src,'to':dst,'kind':kind.value,'confidence':confidence.value,'line':ln,'reason':reason})
    # Map Gradle declared modules to their nested build script; never infer a dependency from colocation.
    # A native target is owned by the *declaring* Gradle build script, not by its name globally.
    cmake_by_build = {}
    module_paths={}
    for path in known:
        if path.endswith(('build.gradle.kts','build.gradle')):
            folder=posixpath.dirname(path)
            name=':' + folder.replace('/',':') if folder else ':root'
            module_paths[name]=path
    for path,text in data.items():
        if path.endswith(('settings.gradle','settings.gradle.kts')):
            for ln,code,original in _lines(text):
                # settings include(":app", ":lib") and Groovy include ':app', ':lib'.
                declared=[]
                if re.search(r'\binclude\s*\(',code):
                    inside=code.split('(',1)[1].split(')',1)[0]
                    declared=re.findall(r'["\'](:[\w.:-]+)["\']',inside)
                elif re.search(r'\binclude\s+',code):
                    inside=re.split(r'\binclude\s+',code,maxsplit=1)[1]
                    declared=re.findall(r'["\'](:[\w.:-]+)["\']',inside)
                for mod in declared:
                    target=module_paths.get(mod)
                    if target: edge(path,target,EdgeType.GRADLE_DEPENDENCY,ConfidenceLevel.EXACT_SYNTAX,ln,original,'declared Gradle include')
                    else: summary['unresolved'].append({'file':path,'line':ln,'kind':'gradle_include','value':mod})
        if path.endswith(('build.gradle','build.gradle.kts')):
            build_lines = list(_lines(text))
            for index, (ln,code,original) in enumerate(build_lines):
                for m in _PROJECT_DEP.finditer(code):
                    target=module_paths.get(m.group(1))
                    if target: edge(path,target,EdgeType.GRADLE_DEPENDENCY,ConfidenceLevel.EXACT_SYNTAX,ln,original,'literal project dependency')
                    else: summary['unresolved'].append({'file':path,'line':ln,'kind':'gradle_project_dependency','value':m.group(1)})
                cm=_CMAKE_PATH.search(code)
                if cm and _cmake_declaration_context(build_lines, index, cm.start()):
                    target=posixpath.normpath(posixpath.join(posixpath.dirname(path),cm.group(1)))
                    if target in known and target.endswith('CMakeLists.txt'):
                        edge(path,target,EdgeType.CMAKE_SOURCE,ConfidenceLevel.EXACT_SYNTAX,ln,original,'declared externalNativeBuild CMake path')
                        cmake_by_build.setdefault(path,set()).add(target)
                if _VARIANT.search(code):
                    summary['variants'].append({'file':path,'line':ln,'text':original,'status':'declared_only_not_evaluated'})
        if path.endswith('pom.xml'):
            try: modules=_maven_modules(text)
            except ET.ParseError as exc:
                summary['unresolved'].append({'file':path,'kind':'invalid_maven_xml','message':str(exc)[:120]});continue
            for module in modules:
                if module.startswith('/') or '..' in Path(module).parts or '${' in module:
                    summary['unresolved'].append({'file':path,'kind':'maven_dynamic_module','value':module});continue
                target=posixpath.normpath(posixpath.join(posixpath.dirname(path),module,'pom.xml'))
                ln=next((i for i,line in enumerate(text.splitlines(),1) if re.search(r'<(?:\w+:)?module\s*>\s*'+re.escape(module)+r'\s*</',line)),1)
                if target in known:
                    edge(path,target,EdgeType.MAVEN_MODULE,ConfidenceLevel.EXACT_SYNTAX,ln,text.splitlines()[ln-1].strip(),'Maven reactor module membership (not dependency)')
                else: summary['unresolved'].append({'file':path,'line':ln,'kind':'maven_module_missing','value':module})
    # A filename match is never enough for native linking. Resolve a literal load
    # only through the nearest declaring Gradle build script and its actual CMake
    # path. The same SHARED target name in another module is irrelevant.
    libraries = {}
    for path, text in data.items():
        if path.endswith('CMakeLists.txt'):
            for ln, code, original in _lines(text):
                # CMake # comments are not C++ comments.
                if code.lstrip().startswith('#'):
                    continue
                for m in _LIBRARY.finditer(code):
                    if m.group(2).upper() == 'SHARED':
                        libraries.setdefault(path, []).append((m.group(1), ln, original))
    build_scripts = sorted((p for p in data if p.endswith(('build.gradle', 'build.gradle.kts'))),
                           key=lambda p: (-len(posixpath.dirname(p).split('/')), p))
    def owner_script(src):
        # Exact module boundary determined by the closest ancestor build script.
        owners = [p for p in build_scripts if not posixpath.dirname(p)
                  or src.startswith(posixpath.dirname(p) + '/')]
        if not owners:
            return None
        return max(owners, key=lambda p: len(posixpath.dirname(p)))

    for path, text in data.items():
        if not path.endswith(('.kt', '.java')):
            continue
        for ln, code, original in _lines(text):
            for m in _LOADLIB.finditer(code):
                if not _not_inside_string(code, m.start()):
                    continue
                name = m.group(1)
                build = owner_script(path)
                declared = sorted(cmake_by_build.get(build, ())) if build else []
                matches = [(cmake, line, snippet) for cmake in declared
                           for target, line, snippet in libraries.get(cmake, ()) if target == name]
                if len(matches) == 1:
                    cmake, target_ln, target_src = matches[0]
                    edge(path, cmake, EdgeType.NATIVE_LIBRARY_LOAD, ConfidenceLevel.CANDIDATE,
                         ln, original, 'loadLibrary target via owning Gradle declared CMake path; runtime/variant unverified')
                    summary.setdefault('native_load_evidence', []).append({
                        'source': path, 'source_line': ln, 'module_build': build,
                        'cmake': cmake, 'cmake_line': target_ln,
                        'cmake_declaration': target_src, 'target': name,
                        'status': 'candidate_declared_build_chain',
                    })
                else:
                    summary['unresolved'].append({
                        'file': path, 'line': ln, 'kind': 'native_load_scope_missing_or_ambiguous',
                        'value': name, 'module_build': build,
                        'declared_cmake_paths': declared,
                        'matching_targets_in_scope': len(matches),
                    })
    summary['links'].sort(key=lambda x:(x['from'],x['to'],x['line'],x['kind']))
    return summary
