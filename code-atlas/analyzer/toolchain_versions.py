"""Offline, read-only evidence for declared language/build tool versions.

Never launches compilers, package managers, build scripts or project code.
Neither constraints nor SDK/target versions prove a deployed runtime.  All
reported evidence is taken from explicitly indexed, in-repository metadata.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import tomllib
import xml.etree.ElementTree as ET

from .php_version_info import _allowed_file
from .build_context import _lines
from .android_variants import _mask_quoted

MAX_MANIFESTS = 500
MAX_EVIDENCE = 2000
MAX_FILE_BYTES = 1_500_000
JSON_FILES = {'package.json', 'global.json'}
TOML_FILES = {'Cargo.toml', 'rust-toolchain.toml', 'pyproject.toml'}
TEXT_FILES = {'go.mod', 'go.work', 'rust-toolchain', '.nvmrc', '.node-version', '.python-version',
              'build.gradle.kts', 'build.gradle', 'gradle.properties'}
XML_FILES = {'pom.xml', 'Directory.Build.props', 'Directory.Build.targets'}
PROJECT_FILES = {'.csproj', '.fsproj', '.vbproj'}
ALLOWED = JSON_FILES | TOML_FILES | TEXT_FILES | XML_FILES
TOOL_VERSIONS = re.compile(r'^\s*(nodejs|node|golang|rust|java|python|kotlin)\s+(\S+)\s*(?:#.*)?$')
GO_DIRECTIVE = re.compile(r'^\s*(go|toolchain)\s+(\S+)\s*$')
# Concise Czech explanations for the public report; not security verdicts.
EXPLANATIONS_CS = {
    'engine_compatibility_range': 'Rozsah kompatibility knihovny; neznáme nainstalovaný Node ani runtime.',
    'package_manager_not_node_runtime': 'Verze správce balíčků; není to verze Node.js.',
    'sdk_selection_not_application_runtime': 'Požadavek pro výběr .NET SDK; není to cílový runtime aplikace.',
    'sdk_selection_policy': 'Pravidlo výběru SDK, ne potvrzení, co je nainstalováno.',
    'minimum_supported_rust_compiler': 'Minimální podporovaná verze kompilátoru Rust (MSRV).',
    'language_edition_not_compiler_version': 'Jazyková edice Rust; není to verze rustc.',
    'requested_toolchain_channel_not_installed': 'Deklarovaný kanál Rust toolchainu; instalaci jsme neověřili.',
    'python_compatibility_constraint': 'Deklarovaná kompatibilita s verzemi Pythonu; není to runtime.',
    'minimum_module_language_toolchain_requirement': 'Minimální požadavek modulu Go; nevíme, která verze běží.',
    'workspace_go_directive': 'Verze Go uvedená v pracovní oblasti; neověřený lokální toolchain.',
    'suggested_toolchain_not_installed': 'Doporučený Go toolchain; jeho instalace není ověřena.',
    'local_version_selection_hint_not_installed': 'Lokální soubor pro výběr verze; nejde o důkaz instalace.',
    'declared_build_plugin_version_not_runtime': 'Deklarovaná verze Kotlin Gradle pluginu, ne runtime Kotlin.',
    'declared_compile_toolchain_level_not_actual_jdk': 'Požadovaná úroveň kompilátoru JVM, nikoli verze dostupného JDK.',
    'declared_source_or_bytecode_target_not_runtime': 'Cílová úroveň zdroje nebo bytecode; runtime serveru neznáme.',
    'declared_property_not_resolved_plugin': 'Deklarovaná Gradle vlastnost; nebyl vyhodnocen build model.',
    'declared_maven_property_not_effective_pom_or_runtime': 'Maven vlastnost bez výpočtu effective POM; není to provozní runtime.',
    'declared_target_framework_not_runtime_or_sdk': 'Cílový .NET framework; nainstalované SDK ani runtime jsme nekontrolovali.',
    'declared_source_language_version_not_compiler': 'Deklarovaná verze jazyka; neověřili jsme kompilátor.',
}

VERSION_LITERAL = re.compile(r'^v?\d+\.\d+(?:\.\d+)?(?:\.\d+)?(?:-[A-Za-z0-9.]+)?$')
GRADLE_KOTLIN_VERSION = re.compile(
    r'\b(?:id\s*\(\s*[\'\"]org\.jetbrains\.kotlin\.[\w.-]+[\'\"]\s*\)|'
    r'id\s+[\'\"]org\.jetbrains\.kotlin\.[\w.-]+[\'\"]|'
    r'kotlin\s*\(\s*[\'\"](?:jvm|android|multiplatform|plugin\.serialization)[\'\"]\s*\))'
    r'\s*version\s*(?:\(\s*)?[\'\"]([^\'\"]+)[\'\"]')
JAVA_TOOLCHAIN = re.compile(r'\b(?:jvmToolchain|JavaLanguageVersion\.of)\s*\(\s*(\d+)\s*\)')
JAVA_COMPAT = re.compile(r'^\s*(sourceCompatibility|targetCompatibility)\s*(?:=|\s)\s*(?:JavaVersion\.VERSION_)?([\d_.]+)\b')
DOTNET_TFM = re.compile(r'^(?:net\d+(?:\.\d+)?|netstandard\d+\.\d+|netcoreapp\d+\.\d+)(?:-[\w.]+)?$', re.I)


def _child(parent: ET.Element, name: str) -> list[ET.Element]:
    return [x for x in parent if x.tag.rsplit('}', 1)[-1] == name]


def _text(parent: ET.Element, name: str) -> str:
    nodes = _child(parent, name)
    return (nodes[0].text or '').strip() if nodes else ''


def _basename(path: str) -> str:
    return PurePosixPath(path).name


def _valid(path: str) -> bool:
    if not isinstance(path, str) or not path or '\\' in path or '\x00' in path:
        return False
    p = PurePosixPath(path)
    return not p.is_absolute() and '..' not in p.parts and p.as_posix() == path


def inspect_toolchain_versions(root: Path, indexed_paths: set[str] | list[str]) -> dict:
    """Return evidence from repository metadata only; zero subprocess/network calls."""
    root = root.resolve()
    known = set(indexed_paths)
    result = {
        'schema': 'atlas-toolchain-version-evidence/1',
        'mode': 'offline_read_only_no_tool_or_build_execution',
        'runtime': {'inspected': False, 'note': 'Nespouštíme žádný interpret ani build a nečteme toolchainy mimo repozitář.'},
        'evidence': [], 'unresolved': [], 'summary': {},
        'interpretation': 'Jde o požadavky, cíle a doporučení z konfigurace, NE o nainstalované nástroje či nasazené runtime.',
    }
    evidence = result['evidence']
    issues = result['unresolved']
    selected: set[str] = set()
    parents = {'.'}
    for path in known:
        if not _valid(path):
            # The source index is not a capability to access other files.
            issues.append({'path': str(path)[:160], 'reason': 'unsafe_index_path'})
            continue
        name = _basename(path)
        if name in ALLOWED or Path(name).suffix in PROJECT_FILES:
            selected.add(path)
            if name in {'package.json', 'Cargo.toml', 'go.mod', 'pom.xml', 'pyproject.toml',
                        'build.gradle', 'build.gradle.kts'}:
                parents.add(str(PurePosixPath(path).parent))
    # Discover only conventional sidecars immediately alongside indexed manifests;
    # do NOT search for tools/files elsewhere on the computer.
    for parent in parents:
        for name in ('.nvmrc', '.node-version', '.python-version', '.tool-versions', 'rust-toolchain',
                     'rust-toolchain.toml', 'global.json', 'gradle.properties'):
            path = (PurePosixPath(parent) / name).as_posix()
            if _valid(path) and (root / path).exists():
                selected.add(path)
    # Limit adversarial repositories with very many manifests.
    if len(selected) > MAX_MANIFESTS:
        issues.append({'reason': 'manifest_limit', 'available': len(selected), 'limit': MAX_MANIFESTS})
    for path in sorted(selected)[:MAX_MANIFESTS]:
        if len(evidence) >= MAX_EVIDENCE:
            issues.append({'reason': 'evidence_limit', 'limit': MAX_EVIDENCE});break
        p = _allowed_file(root, path)
        if p is None:
            issues.append({'path': path, 'reason': 'not_regular_in_scope_or_too_large'})
            continue
        name = _basename(path)
        try:
            raw = p.read_bytes()
            if len(raw) > MAX_FILE_BYTES:
                issues.append({'path': path, 'reason': 'manifest_too_large'});continue
            data_text = raw.decode('utf-8-sig')
        except (OSError, UnicodeError):
            issues.append({'path': path, 'reason': 'unreadable_metadata'});continue
        def add(language: str, kind: str, value: str, interpretation: str, line: int | None = None):
            if not isinstance(value, str) or not value.strip():return
            if len(evidence) >= MAX_EVIDENCE:return
            v = value.strip()
            if len(v) > 160:
                issues.append({'path':path, 'kind':kind, 'reason':'oversized_version_value'});return
            evidence.append({'path': path, 'language': language, 'kind': kind, 'version': v,
                             'line_hint': line, 'meaning': interpretation,
                             'explanation_cs': EXPLANATIONS_CS.get(interpretation,'Deklarace v projektu, nikoli zjištěná verze běžícího nástroje.'),
                             'evidence_type': 'declared_metadata_not_installed_or_deployed_runtime'})
        def line_for(value: str) -> int | None:
            return next((i for i,ln in enumerate(data_text.splitlines(),1) if value in ln),None)
        try:
            if name == 'package.json':
                data = json.loads(data_text)
                if not isinstance(data, dict):raise ValueError('invalid_root')
                engines = data.get('engines', {})
                if isinstance(engines, dict):
                    for engine,lang in (('node','Node.js'),('npm','npm'),('yarn','Yarn'),('pnpm','pnpm'),('bun','Bun')):
                        if isinstance(engines.get(engine),str):
                            add(lang,'package_engines_'+engine,engines[engine], 'engine_compatibility_range',line_for(engines[engine]))
                pm = data.get('packageManager')
                if isinstance(pm,str) and re.fullmatch(r'[a-zA-Z][\w-]*@[^\s]{1,100}',pm):
                    add('JavaScript/TypeScript','package_manager_pin',pm,'package_manager_not_node_runtime',line_for(pm))
            elif name == 'global.json':
                data=json.loads(data_text)
                sdk=data.get('sdk') if isinstance(data,dict) else None
                if isinstance(sdk,dict) and isinstance(sdk.get('version'),str):
                    add('.NET','dotnet_sdk_version',sdk['version'],'sdk_selection_not_application_runtime',line_for(sdk['version']))
                    if isinstance(sdk.get('rollForward'),str):
                        add('.NET','dotnet_sdk_roll_forward',sdk['rollForward'],'sdk_selection_policy',line_for(sdk['rollForward']))
            elif name in ('Cargo.toml','rust-toolchain.toml','pyproject.toml'):
                data=tomllib.loads(data_text)
                if name == 'Cargo.toml':
                    package=data.get('package',{})
                    if isinstance(package,dict):
                        if isinstance(package.get('rust-version'),str):
                            add('Rust','cargo_rust_version',package['rust-version'],'minimum_supported_rust_compiler',line_for(package['rust-version']))
                        if isinstance(package.get('edition'),str):
                            add('Rust','cargo_edition',package['edition'],'language_edition_not_compiler_version',line_for('edition'))
                elif name == 'rust-toolchain.toml':
                    tc=data.get('toolchain')
                    if isinstance(tc,dict) and isinstance(tc.get('channel'),str):
                        add('Rust','rustup_toolchain_channel',tc['channel'],'requested_toolchain_channel_not_installed',line_for(tc['channel']))
                else:
                    project=data.get('project')
                    if isinstance(project,dict) and isinstance(project.get('requires-python'),str):
                        add('Python','pyproject_requires_python',project['requires-python'], 'python_compatibility_constraint',line_for(project['requires-python']))
            elif name in ('go.mod','go.work'):
                # Go directives are line-oriented. Never call go mod download.
                for n,ln in enumerate(data_text.splitlines(),1):
                    code=ln.split('//',1)[0].strip()
                    m=GO_DIRECTIVE.fullmatch(code)
                    if m:
                        typ,val=m.groups()
                        if typ=='go' and re.fullmatch(r'1\.\d+(?:\.\d+)?(?:rc\d+|beta\d+)?',val):
                            add('Go','go_directive',val,'minimum_module_language_toolchain_requirement' if name=='go.mod' else 'workspace_go_directive',n)
                        elif typ=='toolchain' and re.fullmatch(r'go1\.\d+(?:\.\d+)?(?:rc\d+|beta\d+)?|default',val):
                            add('Go','go_toolchain_directive',val,'suggested_toolchain_not_installed',n)
            elif name in ('rust-toolchain','.nvmrc','.node-version','.python-version'):
                value=next((line.strip() for line in data_text.splitlines() if line.strip() and not line.lstrip().startswith('#')),'')
                if value and not re.search(r'[\s\x00\r\n]',value):
                    language,kind={'rust-toolchain':('Rust','rustup_toolchain_channel'),'.nvmrc':('Node.js','node_version_hint'),
                                   '.node-version':('Node.js','node_version_hint'),'.python-version':('Python','python_version_hint')}[name]
                    add(language,kind,value,'local_version_selection_hint_not_installed',line_for(value))
            elif name == '.tool-versions':
                for n,line in enumerate(data_text.splitlines(),1):
                    m=TOOL_VERSIONS.fullmatch(line)
                    if m:
                        tool,val=m.groups()
                        if tool=='golang':lang='Go'
                        elif tool=='rust':lang='Rust'
                        elif tool=='java':lang='Java'
                        elif tool=='kotlin':lang='Kotlin'
                        elif tool in ('node','nodejs'):lang='Node.js'
                        else:lang='Python'
                        add(lang,'asdf_tool_version_hint',val,'local_version_selection_hint_not_installed',n)
            elif name in ('build.gradle','build.gradle.kts'):
                for n,code,rawline in _lines(data_text):
                    no_strings = _mask_quoted(code)
                    for m in GRADLE_KOTLIN_VERSION.finditer(code):
                        if not no_strings[m.start()].strip():continue
                        add('Kotlin','gradle_kotlin_plugin_version',m.group(1),'declared_build_plugin_version_not_runtime',n)
                    for m in JAVA_TOOLCHAIN.finditer(no_strings):
                        typ = 'gradle_kotlin_jvm_toolchain' if 'jvmToolchain' in m.group(0) else 'gradle_java_toolchain'
                        add('JVM',typ,m.group(1),'declared_compile_toolchain_level_not_actual_jdk',n)
                    m=JAVA_COMPAT.match(no_strings)
                    if m:
                        add('Java',m.group(1),m.group(2),'declared_source_or_bytecode_target_not_runtime',n)
            elif name == 'gradle.properties':
                for n,line in enumerate(data_text.splitlines(),1):
                    if line.lstrip().startswith(('#','!')):continue
                    m=re.fullmatch(r'\s*kotlin\.version\s*=\s*(\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?)\s*',line)
                    if m:add('Kotlin','gradle_property_kotlin_version',m.group(1),'declared_property_not_resolved_plugin',n)
            elif name in XML_FILES or Path(name).suffix in PROJECT_FILES:
                if '<!DOCTYPE' in data_text.upper() or '<!ENTITY' in data_text.upper():
                    raise ValueError('xml_doctype_or_entity_prohibited')
                rootxml=ET.fromstring(data_text)
                if name=='pom.xml':
                    for props in _child(rootxml,'properties'):
                        for child in props:
                            kind=child.tag.rsplit('}',1)[-1]
                            if kind in ('maven.compiler.release','maven.compiler.source','maven.compiler.target',
                                        'java.version','kotlin.version') and child.text and child.text.strip():
                                language='Kotlin' if kind=='kotlin.version' else 'Java'
                                add(language,'maven_property_'+kind,child.text.strip(),
                                    'declared_maven_property_not_effective_pom_or_runtime',line_for(child.text.strip()))
                else:
                    # csproj TargetFramework(s) applies to the application/library,
                    # not the SDK version. Evaluate neither MSBuild imports nor conditions.
                    for pg in _child(rootxml,'PropertyGroup'):
                        for key in ('TargetFramework','TargetFrameworks','LangVersion'):
                            val=_text(pg,key)
                            if not val or '$(' in val:continue
                            if key in ('TargetFramework','TargetFrameworks'):
                                parts=val.split(';')
                                if not all(DOTNET_TFM.fullmatch(x.strip()) for x in parts):continue
                                for x in parts:add('.NET','msbuild_'+key,x,'declared_target_framework_not_runtime_or_sdk',line_for(x))
                            elif re.fullmatch(r'[\w.]+',val):
                                add('C# / F# / VB','msbuild_language_version',val,
                                    'declared_source_language_version_not_compiler',line_for(val))
        except (ValueError,TypeError,ET.ParseError,RecursionError):
            issues.append({'path':path,'reason':'malformed_or_unsupported_metadata'})
    evidence.sort(key=lambda x:(x['language'],x['path'],x['kind'],x['version']))
    by_language = {}
    for item in evidence:
        by_language[item['language']]=by_language.get(item['language'],0)+1
    result['summary']={'declarations':len(evidence),'languages':by_language,'unresolved':len(issues),
                       'runtime_unknown':True, 'installed_toolchains_not_inspected':True}
    return result
