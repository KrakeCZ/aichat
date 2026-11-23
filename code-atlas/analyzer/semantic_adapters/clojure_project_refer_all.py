"""Conservative JVM :require :refer :all project-source candidate expansion.

Do not load/execute any project namespace or macro, or enumerate runtime Vars.
Only literal, unique, public declarations in a unique .clj source are eligible.
Every candidate remains unverified compiler/runtime name resolution. A missing
resource, classpath conflict, duplicate or dynamic namespace is not a proof.
"""
from __future__ import annotations

MAX_REFER_ALL_LIBS = 32
MAX_PUBLIC_DECLS = 1024


def materialize(files, *, namespace, definition, visibility, libspec_issue, source_platform):
    """Extend per-file explicit refer maps in place; return audit summaries."""
    # Exact JVM .clj only. Shared .cljc can have different exports and the
    # static source projection must never be silently chosen for the JVM.
    namespace_files = {}
    for path, record in files.items():
        ns = record['namespace'][0]
        if ns:
            namespace_files.setdefault(ns[0], []).append((path, record))

    def exports(ns):
        choices = namespace_files.get(ns, ())
        if len(choices) != 1 or source_platform(choices[0][0]) != 'clj':
            return None, 'refer_all_target_not_unique_jvm_source'
        path, record = choices[0]
        found = set()
        listed = []
        for form in record['forms']:
            identified = definition(form)
            if not identified:
                continue
            name, kind, _ = identified
            if name in found:
                return None, 'refer_all_ambiguous_project_declaration'
            found.add(name)
            if visibility(form)[0] != 'public':
                continue
            listed.append(name)
            if len(listed) > MAX_PUBLIC_DECLS:
                return None, 'refer_all_project_export_limit'
        # An empty source export is not evidence that a dynamically loaded
        # namespace exports nothing. This is a source-only candidate list.
        return tuple(sorted(listed)), None

    summaries = {}
    for path, record in files.items():
        observations = []
        if source_platform(path) != 'clj':
            continue
        forms = [f for f in record['forms'] if f.kind == 'list' and
                 len(f.children) >= 2 and f.children[0].kind == 'atom' and
                 f.children[0].value == 'ns']
        if len(forms) != 1:
            continue
        ns_form = forms[0]
        requested = []
        for clause in ns_form.children[2:]:
            if clause.kind != 'list' or not clause.children or clause.children[0].kind != 'atom' or clause.children[0].value != ':require':
                continue
            for lib in clause.children[1:]:
                if lib.kind != 'vector' or not lib.children or libspec_issue(lib, 'clj') is not None:
                    continue
                target = lib.children[0]
                if target.kind != 'atom':
                    continue
                options = lib.children[1:]
                for index in range(0, len(options) - 1, 2):
                    key, val = options[index], options[index + 1]
                    if key.kind == 'atom' and key.value == ':refer' and val.kind == 'atom' and val.value == ':all':
                        requested.append((target.value, lib, options))
                        break
        if not requested:
            continue
        mapping, limitations = record['referred_names']
        inserted = {}
        if len(requested) > MAX_REFER_ALL_LIBS:
            observations.append({'status': 'unavailable', 'reason': 'refer_all_libspec_limit'})
            record['refer_all_evidence'] = observations
            continue
        # Explicit local names can be derived from :refer [x] even if the
        # target source is missing. Preserve any existing conflict blocker.
        for target, spec, options in requested:
            rename = {}
            for i in range(0, len(options) - 1, 2):
                key, val = options[i], options[i + 1]
                if key.kind != 'atom' or key.value != ':rename':
                    continue
                if val.kind != 'map' or len(val.children) % 2:
                    rename = None
                    break
                for j in range(0, len(val.children), 2):
                    src, dst = val.children[j:j + 2]
                    if src.kind != 'atom' or dst.kind != 'atom':
                        rename = None
                        break
                    rename[src.value] = dst.value
                if rename is None:
                    break
            if rename is None:
                observations.append({'namespace': target, 'status': 'unavailable', 'reason': 'refer_all_rename_not_literal'})
                continue
            names, reason = exports(target)
            if names is None:
                observations.append({'namespace': target, 'status': 'unavailable', 'reason': reason})
                continue
            # Clojure :rename may only rename a name available from target;
            # when the source doesn't prove this, refuse the whole expansion.
            if any(old not in names for old in rename):
                observations.append({'namespace': target, 'status': 'unavailable', 'reason': 'refer_all_rename_target_not_in_public_source'})
                continue
            candidates = {rename.get(name, name): name for name in names}
            if len(candidates) != len(names):
                observations.append({'namespace': target, 'status': 'unavailable', 'reason': 'refer_all_rename_collision'})
                continue
            conflicts = 0
            for local, original in candidates.items():
                if local in mapping:
                    mapping[local] = None
                    inserted.pop(local, None)
                    conflicts += 1
                else:
                    mapping[local] = (target, original)
                    inserted[local] = (target, original)
            observations.append({'namespace': target, 'status': 'source_candidates_only',
                                 'candidate_count': len(candidates), 'conflicting_local_names': conflicts,
                                 'verified_runtime_exports': False,
                                 'compiler_resolver_invoked': False,
                                 'classpath_selection_deferred_to_namespace_target': True})
        record['refer_all_materialized_bindings'] = inserted
        record['referred_names'] = (mapping, sorted(set(
            list(limitations) + ['refer_all_project_source_only_not_runtime_ns_publics'])))
        record['refer_all_evidence'] = observations
        summaries[path] = observations
    return summaries
