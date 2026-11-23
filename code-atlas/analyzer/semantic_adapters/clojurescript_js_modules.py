"""Bounded read-only evidence for literal ClojureScript JS module imports.

This is *not* the ClojureScript/Closure/webpack resolver.  It inventories the
literal string libspec and checks optional files and package manifests without
executing JS, package managers, project macros, or compiler configuration.
Changing :target, :package-json-resolution, :foreign-libs, :npm-deps or a
bundler can change the selected module; no candidate is ever a resolved edge.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

MAX_SITES = 256
MAX_SPECIFIER = 240
MAX_PACKAGE_JSON = 65536
MAX_JS_FILE = 4_000_000
_IDENT = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9._-]*$')


def _same_root(root: Path, target: Path) -> bool:
    try:
        return target.resolve(strict=False).is_relative_to(root.resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        return False


def _read_file(root: Path, relative: str, limit: int):
    """Conservatively reject symlinks, non-files, escapes and changed files."""
    if (not relative or '\\' in relative or '\x00' in relative or
        relative.startswith('/') or any(x in ('', '.', '..') for x in relative.split('/'))):
        return None, 'unsafe_relative_resource_path'
    target = root.joinpath(*relative.split('/'))
    if not _same_root(root, target):
        return None, 'outside_project_or_symlink_path'
    try:
        current = root
        for part in relative.split('/'):
            current = current / part
            if current.is_symlink():
                return None, 'symlink_resource_path'
        with target.open('rb') as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                return None, 'nonregular_or_oversized_resource'
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
        if (len(data) != before.st_size or len(data) > limit or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            return None, 'resource_changed_while_reading'
        # Prevent reporting file bytes from a replaced directory entry after read.
        now = target.stat()
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return None, 'resource_changed_while_reading'
        return data, None
    except (OSError, ValueError, RuntimeError):
        return None, 'resource_not_present_or_unreadable'


def _pairs_unique(pairs):
    values = {}
    for k,v in pairs:
        if k in values:
            raise ValueError('duplicate_json_key')
        values[k] = v
    return values


def _package_parts(name: str):
    if name.startswith('@'):
        parts = name.split('/')
        if len(parts) < 2 or not parts[0][1:] or not parts[1]:
            return None
        package = '/'.join(parts[:2])
        suffix = parts[2:]
    else:
        parts = name.split('/')
        package = parts[0]
        suffix = parts[1:]
    package_tokens = package.split('/')
    if package.startswith('@'):
        package_tokens[0] = package_tokens[0][1:]
    if any(not _IDENT.fullmatch(part) for part in (package_tokens + suffix)):
        return None
    return package, suffix


def _source_entry(spec, source_path, root):
    result = {
        'module_specifier': spec,
        'resolution': 'source_literal_only_not_cljs_compiler_or_bundler_resolution',
        'compiler_module_binding_confirmed': False,
        'js_exports_confirmed': False,
        'compiler_target_or_options_confirmed': False,
    }
    if spec.startswith(('./', '../')):
        result['category'] = 'relative_js_module_candidate'
        # Source-relative location is an *inspection candidate*, not a claim
        # about how the active compiler interprets JS module paths.
        if not spec.endswith(('.js','.mjs','.cjs','.json')):
            result['inspection'] = 'specifier_extension_or_resolution_unknown'
            return result
        source_dir = PurePosixPath(source_path).parent
        components = list(source_dir.parts)
        for part in spec.split('/'):
            if part in ('','.'): continue
            if part == '..':
                if not components:
                    result['inspection'] = 'outside_analyzed_root'
                    return result
                components.pop()
            else:
                components.append(part)
        relative = '/'.join(components)
        content, error = _read_file(root, relative, MAX_JS_FILE)
        result['inspection'] = error or 'local_source_relative_file_observed_not_compiler_selected'
        if content is not None:
            result['local_candidate'] = {'path':relative,'sha256':sha256(content).hexdigest(),
                                         'size_bytes':len(content)}
        return result
    if spec.startswith(('/', '#')) or ':' in spec or '\\' in spec or any(ord(c)<32 for c in spec):
        result['category'] = 'unsupported_or_unsafe_specifier'
        result['inspection'] = 'unsupported_literal_module_scheme_or_path'
        return result
    parsed = _package_parts(spec)
    if parsed is None:
        result['category'] = 'unsupported_or_unsafe_specifier'
        result['inspection'] = 'invalid_or_ambiguous_package_name'
        return result
    package, subpath = parsed
    result['category'] = 'npm_package_declaration_candidate'
    result['package_name'] = package
    result['package_subpath'] = '/'.join(subpath) or None
    manifest_path = 'node_modules/' + package + '/package.json'
    data, error = _read_file(root, manifest_path, MAX_PACKAGE_JSON)
    result['inspection'] = error or 'package_metadata_observed_not_module_resolution'
    if data is None:
        return result
    try:
        info = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if type(info) is not dict:
            raise ValueError('not object')
    except (UnicodeError, ValueError, RecursionError):
        result['inspection'] = 'package_json_invalid_or_ambiguous'
        return result
    result['package_manifest'] = {'path':manifest_path, 'sha256':sha256(data).hexdigest()}
    if 'exports' in info:
        result['inspection'] = 'conditional_package_exports_unresolved'
        return result
    if subpath:
        result['inspection'] = 'package_subpath_resolution_unverified'
        return result
    selections=[]
    for field in ('browser','module','main'):
        raw = info.get(field)
        if type(raw) is not str or not raw or len(raw)>MAX_SPECIFIER:
            continue
        value = raw.removeprefix('./')
        if not value.endswith(('.js','.mjs','.cjs')):
            continue
        path = 'node_modules/' + package + '/' + value
        blob, issue = _read_file(root,path,MAX_JS_FILE)
        selections.append({'package_field':field,'candidate_path':path if blob is not None else None,
                           'inspection':issue or 'literal_package_field_file_observed',
                           'sha256':sha256(blob).hexdigest() if blob is not None else None})
    result['package_entry_candidates'] = selections
    return result



def _literal_js_libspec_shape(item):
    """Only examine recognized literal libspec forms, without pretending validity."""
    opts = item.children[1:]
    if len(opts) % 2:
        return False
    allowed = {':as', ':as-alias', ':default', ':refer', ':rename'}
    seen = set()
    for i in range(0, len(opts), 2):
        key, value = opts[i:i+2]
        if key.kind != 'atom' or key.value not in allowed or key.value in seen:
            return False
        seen.add(key.value)
        if key.value in (':as', ':as-alias', ':default'):
            if value.kind != 'atom' or not _IDENT.fullmatch(value.value):
                return False
        elif key.value == ':refer':
            if value.kind != 'vector' or not value.children or any(
                    f.kind != 'atom' or not _IDENT.fullmatch(f.value) for f in value.children):
                return False
        elif key.value == ':rename':
            if value.kind != 'map' or len(value.children) % 2:
                return False
    return True


def analyze_cljs_module_imports(forms, *, source_path: str, root: Path):
    """Inspect literal ns :require string libspecs; fail closed on site limit."""
    import_forms=[]
    ns = [f for f in forms if f.kind=='list' and f.children and
          f.children[0].kind=='atom' and f.children[0].value=='ns']
    if len(ns)!=1:
        return {'status':'namespace_unavailable_or_ambiguous','sites':[], 'site_count':0}
    for clause in ns[0].children[2:]:
        if (clause.kind!='list' or not clause.children or
            clause.children[0].kind!='atom' or clause.children[0].value!=':require'):
            continue
        for entry in clause.children[1:]:
            if entry.kind!='vector' or not entry.children or entry.children[0].kind!='string':
                continue
            import_forms.append(entry)
            if len(import_forms)>MAX_SITES:
                return {'status':'bounded_evidence_unavailable_site_limit', 'sites':[],
                        'site_count':0,'limits':{'max_sites':MAX_SITES}}
    sites=[]
    for item in import_forms:
        literal=item.children[0]
        # Clojure string tokenizer retains source escape sequences rather than
        # unescaping; refuse strings with escapes instead of guessing their name.
        spec=literal.value
        site={'line':literal.line,'column':literal.col,
              'end_line':literal.end_line,'end_column':literal.end_col,
              'source_platform':'cljs'}
        if not _literal_js_libspec_shape(item):
            site.update({'resolution':'unavailable_unsupported_or_invalid_libspec',
                         'compiler_module_binding_confirmed':False})
        elif ('\\' in spec or len(spec)>MAX_SPECIFIER or not spec or
              any(ord(c)<32 for c in spec)):
            site.update({'resolution':'unavailable_escaped_or_invalid_literal',
                         'compiler_module_binding_confirmed':False})
        else:
            site.update(_source_entry(spec,source_path,root))
        sites.append(site)
    return {'status':'literal_js_module_inventory_not_compiler_resolution',
            'site_count':len(sites),'sites':sites,'limits':{'max_sites':MAX_SITES},
            'limitations':['no_closure_module_compilation','no_js_or_npm_execution',
                'no_package_manager_or_network','no_build_target_or_bundler_proof',
                'no_compiler_resolved_module_or_export_binding']}
