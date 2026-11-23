"""Bounded declared tools.deps project-path evidence, never an effective basis.

Reads only nearest in-repository deps.edn data. Selected CLI aliases are
explicitly requested path-only WHAT-IF SCENARIOS; no user/system config, Maven,
Git dependency, local dependency, Clojure CLI, compiler or eval occurs.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

MAX_MANIFEST_BYTES=256_000
MAX_ROOTS=64
MAX_ALIAS_DEPTH=24
MAX_ALIASES=256
MAX_PATH_EXPANSIONS=4096
MAX_RESOURCE_PROBES=30_000
MAX_SELECTED_ALIASES=16
_ALIAS_SEGMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_.*+!?<>-]*(?:/[A-Za-z_][A-Za-z0-9_.*+!?<>-]*)?\Z')


def normalize_cli_aliases(raw=()):
    """Normalize explicit CLI-style -M:dev:test selection; reject ambiguous input.

    Only a conservative literal EDN keyword subset is accepted. This does not
    attempt to infer aliases from manifests or the user's machine.
    """
    if isinstance(raw,str):
        if not raw: return ()
        if not raw.startswith(':'): raise ValueError('clojure_alias_selection_requires_colon_keywords')
        names=[':'+part for part in raw[1:].split(':')]
    elif isinstance(raw,(tuple,list)):
        names=list(raw)
    else:
        raise ValueError('clojure_alias_selection_must_be_string_or_list')
    if len(names)>MAX_SELECTED_ALIASES:
        raise ValueError('clojure_alias_selection_count_limit')
    for name in names:
        if not isinstance(name,str) or not name.startswith(':') or not _ALIAS_SEGMENT.fullmatch(name[1:]):
            raise ValueError('clojure_alias_selection_unsupported_keyword')
    if len(set(names))!=len(names):
        raise ValueError('clojure_alias_selection_duplicate')
    return tuple(names)


class ContextUnavailable(ValueError):
    pass


def _mapping(form):
    if form.kind!='map':raise ContextUnavailable('expected_keyword_map')
    result={}
    for i in range(0,len(form.children),2):
        key=form.children[i]
        if key.kind!='atom' or not key.value.startswith(':') or key.value.startswith('::'):
            raise ContextUnavailable('expected_literal_keyword_map_key')
        if key.value in result:raise ContextUnavailable('duplicate_config_key')
        result[key.value]=form.children[i+1]
    return result


def _validate_edn(form):
    if form.kind not in {'atom','string','map','vector','list','set','char'} or form.metadata:
        raise ContextUnavailable('unsupported_edn_dispatch_or_metadata')
    if form.kind=='atom' and form.value.startswith('::'):
        raise ContextUnavailable('auto_resolved_keyword_requires_reader_namespace')
    for child in form.children:_validate_edn(child)


def _project_path(root, project, text):
    # Cross-platform inspection must reject Windows drive/UNC paths on Linux.
    if (not text or '\x00' in text or '\\' in text or PurePosixPath(text).is_absolute()
            or PureWindowsPath(text).is_absolute() or PureWindowsPath(text).drive
            or '..' in PurePosixPath(text).parts):
        raise ContextUnavailable('project_path_outside_supported_relative_subset')
    path=project/PurePosixPath(text)
    current=root
    for part in path.parts:
        current=current/part
        if current.is_symlink():raise ContextUnavailable('symlink_project_path')
    if not (root/path).resolve().is_relative_to(root):
        raise ContextUnavailable('project_path_outside_repository')
    if (root/path).exists() and not (root/path).is_dir():
        raise ContextUnavailable('project_source_path_not_directory')
    return path.as_posix()


def _read_context(root, manifest, namespace_names, selected_aliases=()):
    project=manifest.parent
    relative=manifest.as_posix()
    context={'status':'unavailable','manifest':{'path':relative},
        'project_root':project.as_posix(),'declared_roots':[],
        'unselected_path_aliases':[], 'compiled_namespace_resources':{},
        'effective_classpath_resolved':False,'selected_cli_aliases':[],
        'scenario':'project_declared_paths_without_cli_aliases_or_external_basis',
        'external_dependencies_resolved':False,'user_or_system_deps_read':False,
        'alias_selection_is_explicit':bool(selected_aliases),
        'selected_alias_paths_only_scenario':bool(selected_aliases)}
    try:
        source=root/manifest
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ContextUnavailable('unsafe_deps_manifest')
        if source.stat().st_size>MAX_MANIFEST_BYTES:
            raise ContextUnavailable('deps_manifest_size_limit')
        raw=source.read_bytes()
        context['manifest']['source_sha256']=hashlib.sha256(raw).hexdigest()
        from .clojure_forms import read_forms
        from .clojure_shape import _decode_string
        forms=read_forms(raw.decode('utf-8-sig'))
        if len(forms)!=1:raise ContextUnavailable('expected_one_deps_map')
        _validate_edn(forms[0]);data=_mapping(forms[0])
        aliases=_mapping(data[':aliases']) if ':aliases' in data else {}
        if len(aliases)>MAX_ALIASES:raise ContextUnavailable('path_alias_count_limit')
        context['selected_cli_aliases']=list(selected_aliases)
        for key,value in aliases.items():
            if value.kind=='map':
                alias_keys=set(_mapping(value))
                if alias_keys & {':extra-paths',':replace-paths',':paths',':classpath-overrides'}:
                    if key not in selected_aliases:context['unselected_path_aliases'].append(key)
        context['unselected_path_aliases'].sort()
        if selected_aliases:
            for name in selected_aliases:
                if name not in aliases:
                    raise ContextUnavailable('selected_cli_alias_not_declared:'+name)
                if aliases[name].kind!='map':
                    raise ContextUnavailable('selected_cli_alias_not_arg_map:'+name)
        roots=[];expansions=0
        def paths(form, kind, alias_name='', chain=()):
            nonlocal expansions
            expansions+=1
            if expansions>MAX_PATH_EXPANSIONS:raise ContextUnavailable('path_alias_expansion_limit')
            if len(chain)>MAX_ALIAS_DEPTH:raise ContextUnavailable('path_alias_depth_limit')
            if form.kind not in {'vector','list'}:raise ContextUnavailable('expected_path_collection')
            discovered=[]
            for item in form.children:
                expansions+=1
                if expansions>MAX_PATH_EXPANSIONS:raise ContextUnavailable('path_alias_expansion_limit')
                if item.kind=='string':
                    text=_decode_string(item.value)
                    path=_project_path(root,project,text)
                    discovered.append({'path':path,'declared_value':text,
                        'line':item.line,'column':item.col,'path_alias_chain':list(chain),
                        'path_role':kind,'declaring_cli_alias':alias_name or None})
                    if len(discovered)>MAX_ROOTS:raise ContextUnavailable('declared_root_limit')
                elif item.kind=='atom' and item.value.startswith(':'):
                    key=item.value
                    if key in chain:raise ContextUnavailable('cyclic_path_alias')
                    if key not in aliases:raise ContextUnavailable('missing_path_alias')
                    discovered.extend(paths(aliases[key],kind,alias_name,(*chain,key)))
                    if len(discovered)>MAX_ROOTS:raise ContextUnavailable('declared_root_limit')
                else:raise ContextUnavailable('non_literal_path_entry')
            return discovered
        selected_override=None
        selected_extra=None
        if selected_aliases:
            for name in selected_aliases:
                conf=_mapping(aliases[name])
                # Classpath overrides can change what is considered a source
                # and cannot be modeled as project roots without effective basis.
                if ':classpath-overrides' in conf:
                    raise ContextUnavailable('selected_cli_alias_classpath_overrides_require_basis')
                if ':paths' in conf and ':replace-paths' in conf:
                    raise ContextUnavailable('selected_cli_alias_conflicting_replace_paths')
                for key in (':paths',':replace-paths'):
                    if key in conf:selected_override=(name,key,conf[key])
                if ':extra-paths' in conf:
                    selected_extra=(name,conf[':extra-paths'])
        base_form=data.get(':paths')
        if selected_override is not None:
            alias_name,alias_key,form=selected_override
            base_roots=paths(form,alias_key,alias_name,(alias_name,))
            context['paths_evidence']={'line':form.line,'column':form.col,
                                       'source':'selected_cli_alias','alias':alias_name}
        elif base_form is not None:
            base_roots=paths(base_form,':paths')
            context['paths_evidence']={'line':base_form.line,'column':base_form.col,
                                       'source':'project_deps_edn'}
        else:
            raise ContextUnavailable('project_paths_not_declared_effective_defaults_unknown')
        extras=[]
        if selected_extra is not None:
            name,form=selected_extra
            extras=paths(form,':extra-paths',name,(name,))
        # Clojure CLI: explicit extra paths precede base/replace paths. The
        # last selected alias containing each option wins, not a union of all.
        for entry in extras+base_roots:
            if entry['path'] not in {item['path'] for item in roots}:
                roots.append(entry)
            if len(roots)>MAX_ROOTS:raise ContextUnavailable('declared_root_limit')
        if selected_aliases:
            context['scenario']='explicit_selected_cli_alias_project_paths_only_no_effective_basis'
            context['alias_selection_evidence']={'selected_cli_aliases':list(selected_aliases),
                'manifest':context['manifest'],
                'precedence':'last_explicit_alias_for_each_path_option_extra_before_project'}
        # Locally declared in-tree :local/root dependencies can contribute
        # source candidates. They remain candidates: no effective basis,
        # transitive classpath ordering, user config or external deps assumed.
        from .clojure_local_basis import local_roots
        local_deps=local_roots(root,project,data,selected_aliases)
        context['declared_local_dependency_evidence']={
            k:v for k,v in local_deps.items() if k!='roots'}
        if local_deps['status']=='unavailable':
            raise ContextUnavailable(local_deps['reason'])
        for entry in local_deps['roots']:
            if entry['path'] not in {item['path'] for item in roots}:
                roots.append(entry)
            if len(roots)>MAX_ROOTS:
                raise ContextUnavailable('declared_root_limit')
        # Probe only exact __init.class resource names, never walk a classpath.
        # Presence blocks source-only evidence: timestamps/AOT basis are unknown.
        probes=0;compiled={};resources={}
        for namespace in sorted(namespace_names):
            resource=namespace_resource(namespace)
            if resource is None:continue
            for entry in roots:
                for suffix in ('.clj','.cljs','.cljc','__init.class'):
                    probes+=1
                    if probes>MAX_RESOURCE_PROBES:raise ContextUnavailable('namespace_resource_probe_limit')
                    relative_resource=PurePosixPath(entry['path'])/(resource+suffix)
                    target=root/relative_resource
                    # Refuse symlinks at every path component, including a
                    # dangling or cyclic link and an in-tree link. A symlink
                    # is NOT evidence of a source file at this classpath root.
                    # This is a read-only exact-resource probe, not a crawl.
                    component=root
                    for part in relative_resource.parts:
                        component=component/part
                        if component.is_symlink():
                            raise ContextUnavailable('symlink_namespace_resource')
                    if not target.resolve().is_relative_to(root):
                        raise ContextUnavailable('namespace_resource_path_outside_repository')
                    if target.exists() and target.is_file():
                        if suffix=='__init.class':
                            compiled.setdefault(namespace,[]).append(relative_resource.as_posix())
                        else:
                            resources.setdefault(namespace,{}).setdefault(suffix,[]).append(relative_resource.as_posix())
        context.update(status='project_paths_declared',declared_roots=roots,
            compiled_namespace_resources=compiled,_namespace_resources=resources)
    except (ContextUnavailable,ValueError,OSError,UnicodeError,RecursionError,RuntimeError) as error:
        context['reason']=(str(error)[:180] if isinstance(error,ContextUnavailable) else 'deps_or_resource_'+type(error).__name__)
        context['declared_roots']=[]
    return context


def namespace_resource(namespace):
    """Classpath-relative resource convention; no path-like namespace escape."""
    if (not namespace or '/' in namespace or '\\' in namespace or ':' in namespace or
            namespace.startswith('.') or namespace.endswith('.') or '..' in namespace):
        return None
    parts=namespace.replace('-','_').split('.')
    if any(not p or p in {'.','..'} for p in parts):return None
    return '/'.join(parts)


def build_source_contexts(root, source_paths, namespace_names=(), *, selected_aliases=()):
    root=root.resolve();cache={};result={}
    selected_aliases=normalize_cli_aliases(selected_aliases)
    for source in sorted(source_paths):
        folder=PurePosixPath(source).parent;manifest=None
        for directory in (folder,*folder.parents):
            candidate=directory/'deps.edn'
            p=root/candidate
            if p.is_symlink() or p.exists():
                manifest=candidate;break
        if manifest is None:
            result[source]={'status':'unavailable' if selected_aliases else 'not_declared',
                'reason':'selected_cli_aliases_no_project_manifest' if selected_aliases else None,
                'manifest':None,'declared_roots':[],
                'selected_cli_aliases':list(selected_aliases),
                'effective_classpath_resolved':False,'scenario':'indexed_declarations_only'}
            continue
        key=manifest.as_posix()
        if key not in cache:cache[key]=_read_context(root,manifest,namespace_names,selected_aliases)
        common=cache[key]
        if selected_aliases and common.get('status')=='project_paths_declared':
            # A file still belongs in the inventory, but if the explicitly
            # selected scenario excludes its source root its outgoing edges
            # cannot describe that scenario. Do not change the shared cache.
            relative_source=PurePosixPath(source)
            active=any(relative_source.is_relative_to(PurePosixPath(item['path']))
                       for item in common['declared_roots'])
            result[source]={**common,'source_on_selected_cli_paths':active,
                'source_visibility_reason':None if active else 'source_not_on_selected_cli_paths'}
        else:
            result[source]=common
    return result


def select_namespace_paths(context, namespace, platform, paths):
    """Return candidate paths + optional context reason, without filesystem I/O.

    Extension preference is global before root order, matching JVM require and
    cljs.util/ns->source. This resolves only a declared project-path scenario.
    """
    # A validated, explicitly supplied classpath takes precedence over the
    # incomplete project-only path scenario. It can reject a false local edge
    # when an external JAR or directory supplies the first matching resource.
    snapshot = context.get('externally_supplied_classpath') if context else None
    if snapshot:
        record = snapshot['resource_probes'].get(namespace, {})
        # A skipped earlier symlink or a duplicate JAR member can otherwise
        # manufacture a false "winning" source from a later classpath root.
        if record.get('blockers'):
            return [], 'classpath_namespace_resource_ambiguous_or_unsafe'
        entries = record.get('candidates', [])
        extensions = {'clj':('.clj','.cljc'), 'cljs':('.cljs','.cljc')}.get(platform, ())
        if platform=='clj' and any(e['resource'].endswith('__init.class') for e in entries):
            return [], 'classpath_snapshot_compiled_namespace_load_order_unverified'
        for ext in extensions:
            candidates=sorted((e for e in entries if e['resource'].endswith(ext)),
                              key=lambda e:e['position'])
            if not candidates:continue
            first=candidates[0]
            if first['entry_kind']!='directory':
                return [],'external_jar_shadows_project_namespace_source'
            root=Path(snapshot['analyzed_root'])
            source_file=Path(first['file_path'])
            if not source_file.is_relative_to(root):
                return [],'external_source_directory_shadows_project_namespace'
            relative=source_file.relative_to(root).as_posix()
            if relative in paths:return [relative],None
            return [],'classpath_resource_not_indexed_or_mismatched_namespace'
        return [],'namespace_not_present_on_supplied_classpath'
    if not context or context.get('status')=='not_declared':return list(paths),None
    if context.get('status')!='project_paths_declared':
        return [],'project_source_context_unavailable'
    if context.get('source_on_selected_cli_paths') is False:
        return [],'source_not_on_selected_cli_paths'
    if platform=='clj' and namespace in context.get('compiled_namespace_resources',{}):
        return [],'compiled_namespace_requires_load_order_proof'
    resource=namespace_resource(namespace)
    if resource is None:return [],'namespace_resource_name_unsupported'
    extensions={'clj':('.clj','.cljc'),'cljs':('.cljs','.cljc')}.get(platform)
    if extensions is None:return [],'shared_source_platform_not_selected'
    available=set(paths)
    catalog=context.get('_namespace_resources',{}).get(namespace,{})
    for extension in extensions:
        existing=catalog.get(extension,[])
        if existing:
            first=existing[0]
            if first in available:return [first],None
            return [],'declared_resource_not_valid_indexed_namespace'
    return [],'namespace_not_on_declared_project_paths'


def source_path_evidence(context, namespace, platform, selected_path):
    if not context or context.get('status')!='project_paths_declared':return None
    resource=namespace_resource(namespace)
    if resource is None:return None
    for entry in context['declared_roots']:
        for suffix in (('.clj','.cljc') if platform=='clj' else ('.cljs','.cljc')):
            if (PurePosixPath(entry['path'])/(resource+suffix)).as_posix()==selected_path:
                return {'manifest':context['manifest'],'project_root':context['project_root'],
                    'selected_root':entry['path'],'root_declaration':entry,
                    'scenario':context['scenario'],'effective_classpath_resolved':False,
                    'selected_cli_aliases':context.get('selected_cli_aliases',[])}
    return None
