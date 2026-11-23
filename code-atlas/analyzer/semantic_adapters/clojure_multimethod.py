"""Offline Clojure/CLJS multimethod implementation *source evidence*.

A defmethod is an installation site, not a new namespace Var and not a
statically selected dispatch target. Source declarations and declared arities
can be inspected without evaluating the dispatch expression or project code.
"""
from __future__ import annotations
from dataclasses import dataclass
from .clojure_libspecs import simple_symbol
from .clojure_scope import _fn_arities


@dataclass(frozen=True)
class MethodSite:
    name_form: object
    dispatch_form: object
    arities: tuple
    spelling: str


def _valid_target(form):
    if simple_symbol(form):
        return True
    if form.kind != 'atom':
        return False
    bits = form.value.split('/')
    if len(bits) != 2 or not all(bits):
        return False
    from .clojure_forms import Form
    return all(simple_symbol(Form('atom', b, (), form.line, form.col)) for b in bits)


def _unqualified_defmethod_is_core(forms, referred, *, macro_name="defmethod"):
    """Do not interpret a renamed, excluded, or project-overridden macro."""
    from .clojure_forms import _definition, _top, _sym
    if macro_name in referred:
        return False
    if any(definition and definition[0] == macro_name
           for form in forms if (definition := _definition(form))):
        return False
    namespaces = [form for form in forms if _top(form, 'ns')]
    if len(namespaces) != 1:
        return False
    clauses = [clause for clause in namespaces[0].children[2:]
               if _top(clause, ':refer-clojure')]
    if len(clauses) > 1:
        return False
    for clause in clauses:
        items = clause.children[1:]
        if len(items) % 2:
            return False
        seen = set()
        for key_form, val in zip(items[::2], items[1::2]):
            key = _sym(key_form)
            if key in seen or key not in {':exclude', ':only', ':rename'}:
                return False
            seen.add(key)
            if key in {':exclude', ':only'}:
                if val.kind not in {'vector', 'list'} or any(not simple_symbol(x) for x in val.children):
                    return False
                names = {x.value for x in val.children}
                if (key == ':exclude' and macro_name in names or
                    key == ':only' and macro_name not in names):
                    return False
            elif key == ':rename':
                if val.kind != 'map' or len(val.children) % 2:
                    return False
                if any(not simple_symbol(x) for x in val.children):
                    return False
                if any(x.value == macro_name for x in val.children[::2]):
                    return False
    return True


def parse_method_site(form, forms, referred, *, platform):
    """Recognize well-shaped literal defmethod only; no macro evaluation."""
    if form.kind != 'list' or len(form.children) < 5:
        return None
    head = form.children[0]
    spelling = head.value if head.kind == 'atom' else ''
    qualified = {'clojure.core/defmethod'} if platform == 'clj' else {'cljs.core/defmethod'} if platform == 'cljs' else {'clojure.core/defmethod', 'cljs.core/defmethod'}
    if spelling not in qualified and not (spelling == 'defmethod' and
                                         _unqualified_defmethod_is_core(forms, referred)):
        return None
    target = form.children[1]
    if not _valid_target(target):
        return None
    arities = _fn_arities(tuple(form.children[3:]))
    if arities is None:
        return None
    return MethodSite(target, form.children[2], tuple(arities), spelling)
