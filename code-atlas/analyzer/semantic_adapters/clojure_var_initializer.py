"""Bounded, source-only `def` / `defonce` initializer partition.

The compiler has distinct grammar for (def name doc-string? init?) and
(defonce name init). We never execute these forms. A malformed declaration
must not leak bogus candidate calls from its would-be initializer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .clojure_forms import Form


@dataclass(frozen=True)
class VariableInitializer:
    forms: tuple['Form', ...]
    status: str
    reason: str | None = None
    has_doc_string: bool = False

    def evidence(self) -> dict:
        result = {
            'status': self.status,
            'initializer_form_count': len(self.forms),
            'literal_grammar': 'def_symbol_optional_doc_and_init_or_defonce_symbol_init',
            'compiler_binding_confirmed': False,
            'runtime_execution_confirmed': False,
            'namespace_load_execution_proven': False,
        }
        if self.has_doc_string:
            result['literal_doc_string_excluded_from_expression_scan'] = True
        if self.reason:
            result['reason'] = self.reason
        if self.forms:
            f=self.forms[0]
            result['initializer_location'] = {'line': f.line, 'column': f.col,
                                               'end_line': f.end_line,
                                               'end_column': f.end_col}
        return result


def parse_variable_initializer(form: 'Form') -> VariableInitializer:
    if form.kind != 'list' or len(form.children) < 2 or form.children[0].kind != 'atom':
        return VariableInitializer((), 'unsupported', 'invalid_literal_variable_form')
    head=form.children[0].value
    if head not in {'def', 'defonce'}:
        return VariableInitializer((), 'unsupported', 'not_literal_def_or_defonce')
    if form.children[1].kind != 'atom':
        return VariableInitializer((), 'unsupported', 'invalid_literal_var_name')
    tail=form.children[2:]
    if head == 'defonce':
        if len(tail) != 1:
            return VariableInitializer((), 'unsupported', 'defonce_requires_single_initializer')
        return VariableInitializer(tuple(tail), 'conditional_defonce_initializer_candidate')
    if not tail:
        return VariableInitializer((), 'no_initializer')
    if len(tail) == 1:
        # (def v "literal") evaluates the string as an initializer; it is
        # NOT documentation unless an additional init form follows it.
        return VariableInitializer(tuple(tail), 'literal_def_initializer_candidate')
    if len(tail) == 2 and tail[0].kind == 'string':
        return VariableInitializer((tail[1],), 'literal_def_initializer_candidate',
                                   has_doc_string=True)
    return VariableInitializer((), 'unsupported', 'invalid_def_doc_string_or_initializer_arity')
