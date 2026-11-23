"""Pure structural parser of Clojure extend-type/extend-protocol declarations.

Extension sites describe source text, never a runtime class, compiler-selected
protocol binding, method implementation or method dispatch decision.  Malformed
or unrecognized shapes yield no partially invented implementation evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from .clojure_libspecs import simple_symbol
from .clojure_multimethod import _valid_target, _unqualified_defmethod_is_core
from .clojure_scope import _fn_arities


@dataclass(frozen=True)
class ProtocolExtensionMethod:
    protocol: object
    type_form: object
    method: object
    arities: tuple
    expansion_macro: str


def parse_extension_methods(form, forms, referred, *, platform):
    if form.kind!='list' or len(form.children)<4:
        return None
    head=form.children[0].value if form.children[0].kind=='atom' else ''
    if head in {'extend-type','extend-protocol'}:
        if not _unqualified_defmethod_is_core(forms,referred,macro_name=head):
            return None
    elif head not in ({'clojure.core/extend-type','clojure.core/extend-protocol'}
                      if platform=='clj' else
                      {'cljs.core/extend-type','cljs.core/extend-protocol'}
                      if platform=='cljs' else
                      {'clojure.core/extend-type','clojure.core/extend-protocol',
                       'cljs.core/extend-type','cljs.core/extend-protocol'}):
        return None
    type_first=head.endswith('extend-type')
    first=form.children[1]
    if not (_valid_target(first) or (not type_first and first.kind=='atom'
                                    and first.value=='nil')):
        return None
    result=[]
    current=None
    for part in form.children[2:]:
        if part.kind=='atom':
            if not (_valid_target(part) or (not type_first and part.value=='nil')):
                return None
            current=part
            continue
        if part.kind!='list' or current is None or len(part.children)<2:
            return None
        method=part.children[0]
        if not simple_symbol(method):
            return None
        arities=_fn_arities(tuple(part.children[1:]))
        if arities is None:
            return None
        proto=current if type_first else first
        typ=first if type_first else current
        if not _valid_target(proto):
            return None
        result.append(ProtocolExtensionMethod(proto,typ,part,tuple(arities),head))
    return tuple(result) if result else None
