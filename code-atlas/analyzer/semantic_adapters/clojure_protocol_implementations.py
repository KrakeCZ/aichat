"""Pure, bounded source extraction for Clojure/CLJS protocol implementation sites.

No Clojure runtime, macroexpansion, class loading or project execution. A
source method is NOT a proven JVM/JS type or selected protocol dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clojure_libspecs import simple_symbol
from .clojure_multimethod import _valid_target, _unqualified_defmethod_is_core
from .clojure_scope import _fn_arities

MAX_IMPLEMENTATIONS_PER_FORM = 256
MAX_REIFY_NESTING = 72


@dataclass(frozen=True)
class SourceImplementation:
    protocol: object
    type_form: object | None
    method: object
    name_form: object
    arities: tuple
    construction: str
    implementation_kind: str
    owner_form: object
    implicit_fields: frozenset[str] = frozenset()
    field_forms: tuple = ()


def _allowed_core_macro(name, form, forms, referred, platform):
    if not form.children or form.children[0].kind != 'atom':
        return False
    spelled = form.children[0].value
    core = 'cljs.core' if platform == 'cljs' else 'clojure.core'
    if spelled == core + '/' + name:
        return True
    # Shared source can include platform-dependent projections; do not mix
    # qualified branches or claim platform selection from the shared reader.
    if platform == 'shared' and spelled in {'cljs.core/' + name, 'clojure.core/' + name}:
        return True
    return spelled == name and _unqualified_defmethod_is_core(
        forms, referred, macro_name=name)


def _method(section, protocol, type_form, construction, owner, fields, field_forms):
    if section.kind != 'list' or len(section.children) < 2:
        return None
    name = section.children[0]
    if not simple_symbol(name):
        return None
    arities = _fn_arities(tuple(section.children[1:]))
    if arities is None:
        return None
    return SourceImplementation(protocol, type_form, section, name,
        tuple(arities), construction, 'direct_source_method', owner, fields,
        tuple(field_forms))


def parse_direct_implementations(form, forms, referred, *, platform):
    """Parse `deftype`, `defrecord` and `reify` literal method groups.

    All-or-nothing validation prevents publishing half of malformed bodies.
    Interface methods are visible too, but no protocol edge may be fabricated.
    """
    if form.kind != 'list' or len(form.children) < 3:
        return None
    head = form.children[0].value if form.children[0].kind == 'atom' else ''
    name = head.rsplit('/', 1)[-1]
    if name not in {'deftype', 'defrecord', 'reify'}:
        return None
    if not _allowed_core_macro(name, form, forms, referred, platform):
        return None
    if name in {'deftype', 'defrecord'}:
        if (len(form.children) < 4 or not simple_symbol(form.children[1])
                or form.children[2].kind != 'vector'):
            return None
        field_forms = form.children[2].children
        if any(not simple_symbol(field) for field in field_forms):
            return None
        field_names = [field.value for field in field_forms]
        if len(field_names) != len(set(field_names)):
            return None
        fields = frozenset(field_names)
        type_form = form.children[1]
        sections = form.children[3:]
    else:
        fields = frozenset()
        field_forms = ()
        type_form = None
        sections = form.children[1:]
    current = None
    found = []
    for section in sections:
        if section.kind == 'atom':
            if not _valid_target(section):
                return None
            current = section
            continue
        if current is None:
            return None
        method = _method(section, current, type_form, name, form, fields, field_forms)
        if method is None:
            return None
        found.append(method)
        if len(found) > MAX_IMPLEMENTATIONS_PER_FORM:
            return None
    return tuple(found) if found else None


def parse_literal_extend(form, forms, referred, *, platform):
    """Read literal `extend` maps without evaluating protocol/map expressions.

    The value expression can be a function symbol, fn form or arbitrary expr.
    No invocation of that value, protocol resolution or type resolution occurs.
    """
    if (form.kind != 'list' or len(form.children) < 4 or
            (len(form.children) - 2) % 2):
        return None
    if not _allowed_core_macro('extend', form, forms, referred, platform):
        return None
    typ = form.children[1]
    if not (_valid_target(typ) or typ.kind == 'atom' and typ.value == 'nil'):
        return None
    found = []
    for protocol, mapping in zip(form.children[2::2], form.children[3::2]):
        if not _valid_target(protocol) or mapping.kind != 'map' or len(mapping.children) % 2:
            return None
        seen = set()
        for name, value in zip(mapping.children[::2], mapping.children[1::2]):
            if name.kind != 'atom' or not name.value.startswith(':') or '/' in name.value:
                return None
            spelling = name.value[1:]
            from .clojure_forms import Form
            candidate = Form('atom', spelling, (), name.line, name.col,
                             (), name.end_line, name.end_col)
            if not simple_symbol(candidate) or spelling in seen:
                return None
            seen.add(spelling)
            found.append(SourceImplementation(protocol, typ, value, candidate,
                (), 'extend', 'literal_method_map_value_candidate', form))
            if len(found) > MAX_IMPLEMENTATIONS_PER_FORM:
                return None
    return tuple(found) if found else None


def walk_reify_sites(form, forms, referred, *, platform):
    """Collect literal nested `reify` candidates while tracking known local scope.

    In unsupported binding macros, fail closed instead of guessing. The method
    site is a source candidate, never proof of a runtime instance.
    """
    from .clojure_scope import binding_names

    def params_and_bodies(parts, scope, depth):
        arities=_fn_arities(tuple(parts))
        if arities is None:
            return
        for params,body in arities:
            names=set()
            for part in params.children:
                if part.kind=='atom' and part.value=='&':
                    continue
                bound=binding_names(part)
                if bound is None:
                    return
                names.update(bound)
            for expression in body:
                yield from scan(expression, scope | names, depth + 1)

    def scan(item, scope, depth):
        if depth > MAX_REIFY_NESTING or item.kind == 'opaque':
            return
        if item.kind == 'list' and item.children:
            head = item.children[0].value if item.children[0].kind == 'atom' else ''
            if head in {'quote','clojure.core/quote','comment','clojure.core/comment'}:
                return
            if head in {'reify','clojure.core/reify','cljs.core/reify'}:
                if head == 'reify' and head in scope:
                    return
                parsed = parse_direct_implementations(item, forms, referred, platform=platform)
                if parsed:
                    yield from parsed
                return
            if head in {'defn','defn-','defmacro'}:
                parts=list(item.children[2:])
                if parts and parts[0].kind=='string':parts.pop(0)
                if parts and parts[0].kind=='map':parts.pop(0)
                if parts and parts[-1].kind=='map':parts.pop()
                yield from params_and_bodies(parts, scope, depth)
                return
            if head in {'fn','fn*','clojure.core/fn','cljs.core/fn'}:
                parts=list(item.children[1:])
                extra=set()
                if parts and parts[0].kind=='atom':
                    extra.add(parts.pop(0).value)
                yield from params_and_bodies(parts, scope | extra, depth)
                return
            if head in {'let','let*','loop','loop*','clojure.core/let',
                        'clojure.core/loop','cljs.core/let','cljs.core/loop'}:
                if len(item.children)<2 or item.children[1].kind!='vector':
                    return
                bindings=item.children[1].children
                if len(bindings)%2:
                    return
                local=set(scope)
                for pattern,value in zip(bindings[::2],bindings[1::2]):
                    yield from scan(value, local,depth+1)
                    bound=binding_names(pattern)
                    if bound is None:
                        return
                    local.update(bound)
                for expression in item.children[2:]:
                    yield from scan(expression,local,depth+1)
                return
            if head in {'letfn','letfn*','if-let','when-let','if-some','when-some',
                        'for','doseq','binding','with-open','as->'}:
                return  # unknown nested lexical scope; no invented reify site
        for child in item.children:
            yield from scan(child,scope,depth+1)
    yield from scan(form,set(),0)
