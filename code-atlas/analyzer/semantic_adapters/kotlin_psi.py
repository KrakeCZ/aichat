"""Kotlin PSI syntax adapter, relying only on local kotlinc compiler libraries.

No app-specific logic. Syntax parsing != compiler symbol binding. Cross-file
references with a single import/same-package candidate remain explicitly
candidate until a genuine semantic resolver is integrated.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from analyzer.symbol_graph import _node_id, _file_id

BRIDGE = Path(__file__).with_name('AtlasKotlinPsiBridge.kt')
BUNDLED_PSI = Path(__file__).resolve().parents[2] / 'vendor' / 'kotlin-psi-1.9.0'


def _locate_java():
    """Locate an *installed* Java runtime, including Android Studio's JBR.

    No project directory is scanned and nothing is installed or executed during
    discovery. A final checked path is required before using this executable.
    """
    env = os.environ
    explicit = env.get('ATLAS_JAVA')
    if explicit:
        p = Path(explicit).expanduser()
        return str(p.resolve()) if p.is_file() else None
    for home in (env.get('ATLAS_JAVA_HOME'), env.get('JAVA_HOME')):
        if home:
            for name in ('java.exe', 'java'):
                p = Path(home).expanduser() / 'bin' / name
                if p.is_file():
                    return str(p.resolve())
    path_java = shutil.which('java')
    if path_java:
        return str(Path(path_java).resolve())
    if os.name == 'nt':
        # Android Studio's private JBR is frequently available even when users
        # did not manually set JAVA_HOME. Do not run Gradle or Android Studio.
        candidates = []
        for base_env in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
            base = env.get(base_env)
            if not base:
                continue
            b = Path(base)
            candidates.extend((
                b / 'Android' / 'Android Studio' / 'jbr' / 'bin' / 'java.exe',
                b / 'Android' / 'Android Studio' / 'jre' / 'bin' / 'java.exe',
                b / 'Programs' / 'Android Studio' / 'jbr' / 'bin' / 'java.exe',
            ))
        for p in candidates:
            if p.is_file():
                return str(p.resolve())
    return None


def _locate_bundled_psi():
    """Validate reproducible shipped Kotlin syntax provider, no compilation.

    Returns (classes, (compiler, stdlib, reflect), problem). A modified or
    incomplete bundle is rejected rather than silently executed.
    """
    manifest = BUNDLED_PSI / 'KOTLIN_PSI_BUNDLE.json'
    if not manifest.is_file():
        return None, None, 'bundled Kotlin PSI provider is absent'
    try:
        data = json.loads(manifest.read_text('utf8'))
        if data.get('schema') != 'atlas-kotlin-psi-bundle/1':
            raise ValueError('invalid bundle schema')
        if hashlib.sha256(BRIDGE.read_bytes()).hexdigest() != data.get('bridge_source_sha256'):
            raise ValueError('trusted bridge source differs from compiled bundle')
        contents = data['files']
        if not isinstance(contents, dict) or len(contents) < 6:
            raise ValueError('incomplete bundled PSI manifest')
        for rel, expected in contents.items():
            if (not isinstance(rel, str) or not rel or '\\' in rel or ':' in rel or
                    any(part in ('', '.', '..') for part in rel.split('/'))):
                raise ValueError('unsafe bundled path')
            p = BUNDLED_PSI.joinpath(*rel.split('/'))
            if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(BUNDLED_PSI.resolve()):
                raise ValueError('missing or redirected PSI file: ' + rel)
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError('PSI bundle integrity mismatch: ' + rel)
        classes = BUNDLED_PSI / 'classes'
        libs = tuple(BUNDLED_PSI / 'lib' / name for name in (
            'kotlin-compiler.jar', 'kotlin-stdlib.jar', 'kotlin-reflect.jar',
            'trove4j.jar', 'annotations-13.0.jar', 'kotlin-script-runtime.jar'))
        if not all(p.is_file() for p in (*libs,
                classes / 'AtlasKotlinPsiBridgeKt.class')):
            raise ValueError('incomplete runtime classpath')
        return classes, libs, None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, None, f'invalid bundled Kotlin PSI provider: {exc}'


def _locate_toolchain():
    """Discover an explicitly configured, local offline Kotlin/Java CLI.

    Android Studio does not guarantee `kotlinc` on PATH. Users may point
    ATLAS_KOTLIN_HOME/KOTLIN_HOME to an existing standalone compiler folder.
    Never search within or invoke tooling shipped by the analyzed project.
    """
    explicit_home = os.environ.get('ATLAS_KOTLIN_HOME') or os.environ.get('KOTLIN_HOME')
    custom_kotlinc = os.environ.get('ATLAS_KOTLINC')
    explicit_java = os.environ.get('ATLAS_JAVA')
    java_home = os.environ.get('JAVA_HOME')

    def executable(candidate):
        if not candidate:
            return None
        p = Path(candidate).expanduser()
        if p.is_file():
            return str(p.resolve())
        return None

    def find_in(home, names):
        if not home:
            return None
        base = Path(home).expanduser()
        for parent in (base / 'bin', base):
            for filename in names:
                found = executable(parent / filename)
                if found:
                    return found
        return None

    bin_names = ('kotlinc.bat', 'kotlinc.cmd', 'kotlinc.exe', 'kotlinc') if os.name == 'nt' else ('kotlinc',)
    kotlin_bin = (executable(custom_kotlinc) or find_in(explicit_home, bin_names)
                  or shutil.which('kotlinc') or shutil.which('kotlinc.bat'))
    java = _locate_java()
    if not kotlin_bin or not java:
        return None, None, None, 'Kotlin PSI requires local kotlinc and Java; specify ATLAS_KOTLIN_HOME and JAVA_HOME (or ATLAS_KOTLINC / ATLAS_JAVA). No project compiler is executed.'

    home = Path(explicit_home).expanduser() if explicit_home and not custom_kotlinc else Path(kotlin_bin).resolve().parent.parent
    libraries = (home / 'lib' / 'kotlin-compiler.jar',
                 home / 'lib' / 'kotlin-stdlib.jar',
                 home / 'lib' / 'kotlin-reflect.jar')
    if not all(p.is_file() for p in libraries):
        # If a PATH launcher is a wrapper/symlink, explicit installed home is
        # still preferred, but never trust a compiler lib from the source root.
        return None, None, None, f'Kotlin PSI compiler libraries not found under {home / "lib"}; set ATLAS_KOTLIN_HOME to the standalone Kotlin distribution.'
    return str(kotlin_bin), str(java), libraries, None


def decode(value: str) -> str:
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)).decode('utf-8')


def analyze_kotlin(root: Path, candidates: dict, _graph_data: dict) -> tuple[dict, list]:
    paths = sorted(p for p, v in candidates.items() if v.get('language') == 'kotlin')
    if not paths:
        return {}, []
    warnings = []
    root = root.resolve()
    # A pinned, precompiled PSI provider is part of the *trusted Atlas app*.
    # It does not require kotlinc during the analysis, and never compiles the
    # analyzed project. External tools are a development fallback only if the
    # trusted bundle is wholly absent (not if its integrity check fails).
    bundled_present = (BUNDLED_PSI / 'KOTLIN_PSI_BUNDLE.json').exists()
    bundle_classes, bundle_libraries, bundle_issue = _locate_bundled_psi()
    use_bundle = bundle_classes is not None
    if use_bundle:
        executable, java, libraries = None, _locate_java(), bundle_libraries
        if not java:
            return {}, [{'reason': 'kotlin_psi_java_unavailable',
                         'details': 'Bundled Kotlin PSI is installed, but Java is missing. Set ATLAS_JAVA or ATLAS_JAVA_HOME, or install Android Studio JBR.'}]
    elif bundled_present:
        return {}, [{'reason':'kotlin_psi_bundle_integrity_failed', 'details':bundle_issue}]
    else:
        executable, java, libraries, issue = _locate_toolchain()
        if issue:
            return {}, [{'reason': 'kotlin_psi_unavailable', 'details': issue}]
    compiler, stdlib, reflect = libraries[:3]
    # Even an explicitly configured runtime must not come from the analyzed
    # project. Bundled PSI classes are trusted Atlas bytes, not project code.
    for role, program in (('kotlinc', executable), ('java', java)):
        if program and Path(program).resolve().is_relative_to(root):
            return {}, [{'reason': 'kotlin_psi_provider_inside_project',
                         'details': f'{role} resolves inside analyzed source tree'}]
    files, raw_files, hashes = [], {}, {}
    for p in paths:
        path = (root / p).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 2_000_000:
            warnings.append({'path': p, 'reason': 'kotlin_file_skipped'}); continue
        raw = path.read_bytes()
        hashes[p] = hashlib.sha256(raw).hexdigest()
        raw_files[p] = raw.decode('utf-8-sig', errors='replace').splitlines()
        files.append(p)
    if not files:
        return {}, warnings
    # The PSI bridge parses each source independently. Process bounded groups to
    # avoid truncating a repository after 500 Kotlin files. Import/name candidates
    # are connected *after* all groups have been parsed, across group boundaries.
    stdout_parts = []
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-kotlin-psi-') as t:
            temp = Path(t)
            listing = temp / 'sources.list'
            classes = temp / 'classes'
            if use_bundle:
                classes = bundle_classes  # verified immutable precompiled bridge
            else:
                # Development fallback: only the Atlas-owned bridge is compiled,
                # NEVER any sources from the analyzed project.
                proc = subprocess.run([executable, '-cp', str(compiler), str(BRIDGE), '-d', str(classes)],
                                      capture_output=True, text=True, timeout=95, cwd=temp)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[:500])
            cp = os.pathsep.join(map(str, (classes, *libraries)))

            def parse_group(group):
                # A huge source file must not cause all valid files in its batch
                # to disappear. Bisect a failed batch until failures are isolated.
                listing.write_text('\n'.join(group) + '\n', encoding='utf8')
                try:
                    result = subprocess.run([java, '-Xmx640m', '-cp', cp,
                                             'AtlasKotlinPsiBridgeKt', str(root), str(listing)],
                                            capture_output=True, text=True, timeout=90, cwd=temp)
                    if result.returncode:
                        raise RuntimeError(result.stderr[:400] or f'exit status {result.returncode}')
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    if len(group) == 1:
                        warnings.append({'path':group[0], 'reason':'kotlin_psi_batch_failed',
                                         'details':str(exc)[:260]})
                    else:
                        middle = len(group) // 2
                        parse_group(group[:middle])
                        parse_group(group[middle:])
                    return
                stdout_parts.append(result.stdout)
                if result.stderr.strip():
                    warnings.append({'reason':'kotlin_psi_partial_diagnostic',
                                     'details':result.stderr[:260], 'first_path':group[0]})

            for offset in range(0, len(files), 128):
                parse_group(files[offset:offset + 128])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {}, warnings + [{'reason': 'kotlin_psi_failed', 'details': str(exc)[:350]}]
    # Preserve batch boundaries: PSI lines are tab-delimited, and each batch
    # ends with newlines. Parsing all records together enables cross-batch links.
    stdout = '\n'.join(stdout_parts)
    graphs = {p: {'nodes': [{'id': _file_id(p), 'path':p, 'name':p.split('/')[-1],
                           'kind':'file', 'line':1, 'confidence':'verified_file'}],
                  'edges': [], 'parser':'kotlin_psi', 'hash':hashes[p],
                  'caution':'Kotlin compiler PSI syntax. Calls are lexical candidates, NOT compiler-resolved dispatch.'}
              for p in files}
    all_nodes = defaultdict(lambda: defaultdict(list))
    package = {}
    imports = defaultdict(list)
    calls = []
    signatures = {}
    complete_files = set()
    rejected_files = {}
    param_types = {}
    extension_receivers = {}
    for record in stdout.splitlines():
        # Individual PSI batches end in a newline. Joining them creates a
        # harmless separator line; it is not a corrupted parser record.
        if not record.strip():
            continue
        try:
            parts = [decode(s) for s in record.split('\t')]
            kind, path = parts[:2]
            if path not in graphs:
                continue
            if kind == 'E' and len(parts)==2:
                complete_files.add(path)
            elif kind == 'R' and len(parts)==3:
                rejected_files[path]=parts[2]
            elif kind == 'P' and len(parts)==3:
                package[path]=parts[2]
            elif kind=='I' and len(parts)==5:
                imports[path].append((parts[2], parts[3], int(parts[4])))
            elif kind=='N' and len(parts)==10:
                _,_,nkind,name,qual,parent,ln,col,end,native = parts[:10]
                line, column = int(ln),int(col)
                node={'id':_node_id(path,qual,line,column),'path':path,'kind':nkind,
                      'name':name,'qualname':qual,'line':line,'column':column,
                      'end_line':int(end),'confidence':'verified_syntax',
                      'external_declaration': native=='true','_parent_qual':parent}
                graphs[path]['nodes'].append(node)
                all_nodes[path][qual].append(node)
            elif kind=='T' and len(parts)==6:
                param_types[(path,parts[2],int(parts[3]),int(parts[4]))] = parts[5].split('\u001f') if parts[5] else []
            elif kind=='S' and len(parts)==7:
                spec = []
                for raw_spec in parts[6].split('\u001e') if parts[6] else []:
                    fields = raw_spec.split('\u001f')
                    if len(fields) != 3 or fields[1] not in ('true', 'false') or fields[2] not in ('true', 'false'):
                        raise ValueError('invalid Kotlin signature')
                    spec.append({'name': fields[0], 'default': fields[1] == 'true',
                                 'vararg': fields[2] == 'true'})
                signatures[(path,parts[2],int(parts[3]),int(parts[4]))] = (parts[5],spec)
            elif kind=='X' and len(parts)==6:
                extension_receivers[(path, parts[2], int(parts[3]), int(parts[4]))] = parts[5]
            elif kind=='C' and len(parts) in (6,9):
                arity = int(parts[6]) if len(parts)==9 else None
                names = parts[7].split('\u001f') if len(parts)==9 and parts[7] else []
                location = parts[8] if len(parts)==9 else ''
                if arity is not None and (arity < 0 or arity < len(names)):
                    raise ValueError('invalid Kotlin argument shape')
                calls.append((path,parts[2],parts[3],int(parts[4]),parts[5],arity,names,location))
        except (ValueError, UnicodeError, IndexError):
            warnings.append({'reason':'kotlin_psi_invalid_record'})
    # The Kotlin bridge can emit partial records before an exception. Do not
    # claim a source file was parsed unless its explicit end record arrived.
    for path in files:
        if path in rejected_files:
            graphs.pop(path, None)
            warnings.append({'path':path,'reason':'kotlin_psi_source_rejected',
                             'details':rejected_files[path][:200]})
        elif path not in complete_files:
            graphs.pop(path, None)
            warnings.append({'path':path,'reason':'kotlin_psi_incomplete_source'})
    for path, g in graphs.items():
        for node in g['nodes']:
            key = (path, node.get('qualname', ''), node.get('line'), node.get('column'))
            if key in param_types and node.get('kind') in ('function','method'):
                node['source_parameter_types'] = param_types[key]
            if key in signatures and node.get('kind') in ('function','method'):
                visibility, parameters = signatures[key]
                node['source_visibility'] = visibility
                node['source_parameters'] = parameters
            if key in extension_receivers and node.get('kind') in ('function','method'):
                node['source_extension_receiver_type'] = extension_receivers[key]
    # Attach verified declaration containment, never assume reference identity.
    def evidence(path, line):
        lines=raw_files[path]
        return {'path':path,'line':line,'source_sha256':hashes[path],
                'snippet':lines[line-1].strip()[:200] if 0<line<=len(lines) else ''}
    for path,g in graphs.items():
        for n in g['nodes'][1:]:
            parent_qual=n.pop('_parent_qual','')
            if '\u001e' in parent_qual:
                try:
                    q, ln, col = parent_qual.split('\u001e')
                    parents = [p for p in all_nodes[path].get(q, [])
                               if p['line'] == int(ln) and p['column'] == int(col)]
                except (ValueError, TypeError):
                    parents = []
            else:
                parents=all_nodes[path].get(parent_qual,[]) if parent_qual else [g['nodes'][0]]
            if len(parents)!=1:continue
            n['parent']=parents[0]['id']
            g['edges'].append({'source':parents[0]['id'],'target':n['id'],
                'source_path':path,'target_path':path,'relation':'declares',
                'resolution':'verified_syntax','analyzer':'kotlin-psi',
                'evidence':evidence(path,n['line'])})
    for path, graph in graphs.items():
        for node in graph['nodes']:
            node['package_name'] = package.get(path, '')
    top=defaultdict(list)
    for path,g in graphs.items():
        for n in g['nodes'][1:]:
            if n['kind']=='function':
                top[(package.get(path,''),n['name'])].append(n)
    # Kotlin PSI is syntactic evidence. The following rules narrow *candidate*
    # sets based on lexical scopes, explicit import priority, visibility, named
    # arguments, defaults, and arity. They do NOT infer types or certify dispatch.
    def applicable(node, count, names):
        parameters = node.get('source_parameters')
        if count is None or parameters is None:  # older PSI bridges: no guess
            return True
        defined = [p['name'] for p in parameters]
        if any(name not in defined for name in names) or len(set(names)) != len(names):
            return False
        positional = count - len(names)
        variadic = next((i for i,p in enumerate(parameters) if p['vararg']), None)
        if variadic is None and positional > len(parameters):
            return False
        if variadic is not None and variadic < len(parameters)-1:
            # Kotlin permits varargs before named-only arguments. Without
            # complete argument mapping, do not reject potentially valid calls.
            return True
        assigned = set(names)
        for index, param in enumerate(parameters[:positional]):
            if variadic is not None and index >= variadic:
                break
            if param['name'] in assigned:
                return False  # duplicate positional + named assignment
            assigned.add(param['name'])
        for param in parameters:
            if param['name'] not in assigned and not param['default'] and not param['vararg']:
                return False
        return True

    def accessible(node, path):
        # Top-level private declarations are visible only from their source file.
        return node['path'] == path or node.get('source_visibility') != 'private'

    def select(group, path, count, names):
        seen = set()
        matches = []
        for node in group:
            # A receiver extension cannot be identified from a bare function
            # name. This source-only pass has no implicit receiver type facts.
            if node.get('source_extension_receiver_type'):
                continue
            if node['id'] not in seen and accessible(node, path) and applicable(node, count, names):
                matches.append(node)
                seen.add(node['id'])
        return matches

    for path,owner,name,line,receiver,count,names,owner_location in calls:
        if path not in graphs:
            continue
        graph = graphs[path]
        if owner:
            owners = all_nodes[path].get(owner, [])
            if owner_location:
                try:
                    ln, col = map(int, owner_location.split('\u001e'))
                    owners = [n for n in owners if n['line'] == ln and n['column'] == col]
                except ValueError:
                    owners = []
        else:
            owners = [graph['nodes'][0]]
        if len(owners) != 1:
            continue
        if receiver not in ('', 'this'):
            continue  # unknown receiver type: cannot select a source member
        targets = []
        # Traverse *actual source parent identities*, not dotted names. Two
        # overloaded parent functions can both contain e.g. helper() and run().
        # Name-only traversal would incorrectly join their different scopes.
        source_nodes = {node['id']: node for node in graph['nodes']}
        ancestors = []
        current = owners[0]
        visited = set()
        while current and current['id'] not in visited:
            visited.add(current['id'])
            ancestors.append(current)
            current = source_nodes.get(current.get('parent'))
        if receiver == '':
            for scope_node in ancestors:
                # Top-level declarations participate in the package/import
                # candidate groups below. An explicit import alias can select
                # a differently named function despite a same-named local.
                if scope_node.get('kind') == 'file':
                    continue
                local = [n for n in graph['nodes'][1:]
                         if n.get('kind') in ('function', 'method') and n['name'] == name
                         and n.get('parent') == scope_node['id']]
                targets = select(local, path, count, names)
                if targets:
                    break
        # Explicit this is restricted to the nearest *actual* containing class;
        # do not join unrelated classes with the same textual name.
        if not targets and receiver == 'this':
            classes = [n for n in ancestors if n.get('kind') in
                       ('class', 'interface', 'enum', 'object')]
            if classes:
                cls = classes[0]
                member = [n for n in graph['nodes'][1:]
                          if n.get('kind') == 'method' and n['name'] == name
                          and n.get('parent') == cls['id']]
                targets = select(member, path, count, names)
        if not targets and receiver == '':
            # Kotlin prioritizes explicit imports over package declarations,
            # and package declarations over star imports.
            explicit = []
            stars = []
            for fq, alias, _ in imports[path]:
                if fq.endswith('.*'):
                    stars.append(fq[:-2]); continue
                if (alias or fq.rsplit('.',1)[-1]) == name:
                    explicit.append(fq)
            groups = []
            # Match the original symbol name, not the imported alias.
            if explicit:
                groups.append([candidate for fq in explicit
                               for candidate in top.get((fq.rpartition('.')[0], fq.rpartition('.')[2]), [])])
            groups.append(top.get((package.get(path,''),name), []))
            groups.append([candidate for pkg in stars for candidate in top.get((pkg,name), [])])
            for group in groups:
                targets = select(group,path,count,names)
                if targets:
                    break
        for target in targets:
            edge = {'source':owners[0]['id'],'target':target['id'],
                'source_path':path,'target_path':target['path'],'relation':'call_candidate',
                'resolution':'psi_name_candidate','analyzer':'kotlin-psi',
                'evidence':evidence(path,line)}
            if count is not None:
                edge['source_argument_count'] = count
                edge['source_named_arguments'] = names
            graph['edges'].append(edge)
    # Externally defined functions used as targets are copied as navigable references.
    known={n['id']:n for g in graphs.values() for n in g['nodes']}
    for g in graphs.values():
        have={n['id'] for n in g['nodes']}
        for e in g['edges']:
            for ref in (e['source'],e['target']):
                if ref not in have and ref in known:
                    g['nodes'].append({**known[ref],'external':True})
                    have.add(ref)
    return graphs,warnings
