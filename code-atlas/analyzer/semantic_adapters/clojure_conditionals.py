"""Conservative, offline .cljc reader-conditional projection, not evaluation.

First-match :clj/:cljs/:default semantics apply inside reader wrappers,
metadata and list/vector/map/set collections. Ordinary quoted/regex/fn-literal
reader wrappers remain OPAQUE: selecting a branch does not make its contents
executable calls, typed expressions, macro expansion or resolved symbols.
Unknown tagged readers still fail closed. Nothing executes project code.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clojure_forms import Form
from .clojure_shape import literal_identity
from .clojure_namespaced_maps import normalize_map, NamespaceMapUnavailable


class ProjectionUnavailable(ValueError):
    """Reason why no exact platform data tree should be claimed."""


@dataclass(frozen=True)
class Projection:
    platform: str
    forms: tuple[Form, ...]
    selected_conditionals: int
    omitted_conditionals: int
    spliced_elements: int


MAX_PROJECTED_NODES = 25_000
MAX_PROJECTED_DEPTH = 80


def project(forms: list[Form] | tuple[Form, ...], platform: str, *,
            preserve_unknown_tagged_syntax: bool = False) -> Projection:
    """Build a bounded *candidate* data tree for a caller-selected platform.

    Features are explicitly selected, never guessed from repository names.
    No namespace/import or runnable expression is evaluated. Return value
    retains original source locations for downstream candidate evidence.
    """
    if platform not in ('clj', 'cljs'):
        raise ProjectionUnavailable('platform must be explicitly clj or cljs')
    counts = {'nodes': 0, 'selected': 0, 'omitted': 0, 'spliced': 0}

    def project_metadata(form: Form, depth: int) -> tuple[Form, ...]:
        """Metadata is reader data, selected separately for each dialect.

        #?(:clj String :cljs js/Object) is a valid metadata TAG. Never
        validate that opaque source tag before platform selection, and never
        smuggle an absent or splicing tag into a completed source projection.
        """
        selected_metadata = []
        for entry in form.metadata:
            expanded = expand(entry, None, depth + 1)
            if len(expanded) != 1:
                raise ProjectionUnavailable('metadata must select exactly one tag')
            tag = expanded[0]
            if tag.kind not in ('map', 'string', 'atom', 'vector'):
                raise ProjectionUnavailable('projected metadata tag invalid')
            if tag.kind == 'atom':
                atom = tag.value
                if atom in ('nil','true','false') or (atom[:1].isdigit() or
                        (atom[:1] in '+-' and len(atom) > 1 and atom[1].isdigit())):
                    raise ProjectionUnavailable('projected metadata tag invalid')
            selected_metadata.append(tag)
        return tuple(selected_metadata)

    def expand(form: Form, container: str | None, depth: int) -> list[Form]:
        if depth > MAX_PROJECTED_DEPTH:
            raise ProjectionUnavailable('conditional projection depth limit')
        counts['nodes'] += 1
        if counts['nodes'] > MAX_PROJECTED_NODES:
            raise ProjectionUnavailable('conditional projection node limit')

        if form.kind == 'opaque' and form.value in ('#?', '#?@'):
            if len(form.children) != 1 or form.children[0].kind != 'list':
                raise ProjectionUnavailable('conditional must contain feature/expression list')
            clauses = form.children[0].children
            selected = None
            # Important JVM behavior: once a branch matches, subsequent tags
            # and values are SKIPPED rather than validated. An unmatched
            # trailing tag without an expression also reads as no form. Only
            # a trailing *matching* tag must fail due to missing expression.
            # Preserve this exact first-match distinction (Stage40 oracle).
            for i in range(0,len(clauses),2):
                feature = clauses[i]
                if (feature.kind != 'atom' or not feature.value.startswith(':') or
                        feature.value.startswith('::') or '/' in feature.value):
                    raise ProjectionUnavailable('unsupported conditional feature tag')
                if feature.value in (':' + platform, ':default'):
                    if i+1 >= len(clauses):
                        raise ProjectionUnavailable('matched feature has no expression')
                    selected = clauses[i+1]
                    break
            if selected is None:
                counts['omitted'] += 1
                return []    # reader skips, *not* a nil form
            counts['selected'] += 1
            if form.value == '#?':
                return expand(selected, container, depth+1)
            if container not in ('list','vector','map','set'):
                raise ProjectionUnavailable('splicing requires a supported parent collection')
            if selected.kind not in ('vector','list'):
                raise ProjectionUnavailable('splicing payload must be sequential')
            result = []
            for child in selected.children:
                result.extend(expand(child, container, depth+1))
            counts['spliced'] += len(result)
            return result

        if form.kind == 'namespaced_map':
            items: list[Form] = []
            for child in form.children:
                items.extend(expand(child, 'map', depth+1))
            if len(items) % 2:
                raise ProjectionUnavailable('projected namespaced map has odd entries')
            candidate = Form('namespaced_map',form.value,tuple(items),form.line,form.col,
                             project_metadata(form,depth),form.end_line,form.end_col)
            try:
                # Auto-resolved map namespaces need the source :ns/:require
                # after the first conditional projection. Resolve in pass two.
                normalized = (None if form.value.startswith('#::') else
                              normalize_map(candidate))
            except NamespaceMapUnavailable as exc:
                raise ProjectionUnavailable(str(exc)) from exc
            return [normalized if normalized is not None else candidate]
        if form.kind == 'opaque':
            # Standard reader wrappers are preserved verbatim and remain
            # uninterpreted. Their *contained reader conditionals* are still
            # selected as specified by the reader, before macro evaluation.
            # Unknown tagged readers cannot be treated as valid code without
            # an opted-in, safe reader registry: remain unavailable.
            if form.value == '#(':
                parts = []
                for child in form.children:
                    parts.extend(expand(child, 'list', depth+1))
                return [Form('opaque', '#(', tuple(parts), form.line, form.col,
                             project_metadata(form,depth),form.end_line,form.end_col)]
            if form.value in ("'", "`", "#'", "@", "~", "~@") and len(form.children) == 1:
                parts=expand(form.children[0], None, depth+1)
                if len(parts) != 1:
                    raise ProjectionUnavailable('reader prefix operand must select one form')
                return [Form('opaque',form.value,tuple(parts),form.line,form.col,
                             project_metadata(form,depth),form.end_line,form.end_col)]
            if (len(form.children) == 1 and form.value.startswith('#') and
                    (preserve_unknown_tagged_syntax or
                     form.value in ('#inst','#uuid') or
                     (platform == 'cljs' and form.value == '#js'))):
                parts = expand(form.children[0], None, depth+1)
                if len(parts) != 1:
                    raise ProjectionUnavailable('reader tag operand selects no single form')
                return [Form('opaque',form.value,tuple(parts),form.line,form.col,
                             project_metadata(form,depth),form.end_line,form.end_col)]
            raise ProjectionUnavailable('unknown tagged or reader dispatch: '+form.value[:20])
        if form.kind in ('list','vector','map','set'):
            items: list[Form] = []
            for child in form.children:
                items.extend(expand(child,form.kind,depth+1))
            if form.kind=='map' and len(items)%2:
                raise ProjectionUnavailable('projected map has odd key/value count')
            if form.kind in ('map','set'):
                members = items[::2] if form.kind=='map' else items
                known = set()
                for item in members:
                    identity=literal_identity(item)
                    if identity is not None:
                        if identity in known:
                            raise ProjectionUnavailable('duplicate projected map key or set value')
                        known.add(identity)
            return [Form(form.kind,form.value,tuple(items),form.line,form.col,
                         project_metadata(form,depth),form.end_line,form.end_col)]
        if form.kind in ('atom','string','char','symbolic_value','regex'):
            if not form.metadata:
                return [form]
            return [Form(form.kind,form.value,form.children,form.line,form.col,
                         project_metadata(form,depth),form.end_line,form.end_col)]
        raise ProjectionUnavailable('unrecognized reader form kind: '+form.kind[:30])

    output = []
    for form in forms:
        output.extend(expand(form,None,0))
    # A two-pass source-only lookup avoids assuming that aliases from one
    # platform apply to the other. No require is performed.
    from .clojure_forms import _namespace, _libspec_issue, _sym, _top
    parsed_ns, aliases, _, issue = _namespace(output,platform)
    # :as-alias is a reader namespace alias without a runtime require. Never
    # turn it into a loaded library, but allow its exact literal mapping to
    # resolve #::alias keys. Duplicate/invalid aliases remain unresolved.
    alias_candidates: dict[str, list[str]] = {}
    for statement in output:
        if not _top(statement,'ns'):
            continue
        for clause in statement.children[2:]:
            if (clause.kind!='list' or not clause.children or
                    _sym(clause.children[0])!=':require'):
                continue
            for spec in clause.children[1:]:
                if spec.kind!='vector' or not spec.children or _libspec_issue(spec,platform):
                    continue
                lib = _sym(spec.children[0])
                for i in range(1,len(spec.children)-1,2):
                    if _sym(spec.children[i]) in (':as',':as-alias'):
                        name = _sym(spec.children[i+1])
                        alias_candidates.setdefault(name,[]).append(lib)
    for alias, targets in alias_candidates.items():
        if len(targets)==1 and alias not in aliases:
            aliases[alias]=targets[0]
        elif len(targets)!=1 or (alias in aliases and aliases[alias]!=targets[0]):
            aliases.pop(alias,None)
    current_ns = parsed_ns[0] if parsed_ns is not None and issue is None else None

    def resolve_auto(form: Form):
        children = tuple(resolve_auto(ch) for ch in form.children)
        metadata = tuple(resolve_auto(m) for m in form.metadata)
        revised = Form(form.kind,form.value,children,form.line,form.col,
                       metadata,form.end_line,form.end_col)
        if revised.kind == 'namespaced_map':
            try:
                resolved = normalize_map(revised,current_namespace=current_ns,
                                         aliases=aliases)
            except NamespaceMapUnavailable as exc:
                raise ProjectionUnavailable(str(exc)) from exc
            return resolved or revised
        return revised
    return Projection(platform,tuple(resolve_auto(f) for f in output),
                      counts['selected'],counts['omitted'],counts['spliced'])
