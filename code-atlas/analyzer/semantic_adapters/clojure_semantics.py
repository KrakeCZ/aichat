"""Offline, non-evaluating lexical and core-special-form semantics for Clojure.

This module deliberately never calls a compiler, macro, build task or data reader.
Its CFG records *syntactic* branch ordering for a strictly recognized subset,
its binding references prove lexical identity only, and Java/JS interop records
syntax without claiming a resolved member or type.  Every unsupported construct
is explicit in diagnostics.  No runtime reachability is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .clojure_scope import binding_names, _identifier, _has_map_default_values, _default_expression_forms
from .clojure_function_scope import function_contract_plan

MAX_FLOW_NODES = 6000
MAX_DIAGNOSTICS = 200


def _head(form):
    return (form.children[0].value if form.kind == 'list' and form.children
            and form.children[0].kind == 'atom' else '')


def _loc(form):
    return {'line': form.line, 'column': form.col,
            'end_line': form.end_line, 'end_column': form.end_col}


def _arity_body(definition):
    """Find only compiler-shaped defn/defmacro arities; metadata/doc is skipped."""
    parts = list(definition.children[2:])
    if parts and parts[0].kind == 'string':
        parts.pop(0)
    if parts and parts[0].kind == 'map':
        parts.pop(0)
    if not parts:
        return [], 'missing_arity'
    if parts[0].kind == 'vector':
        return [(parts[0], tuple(parts[1:]))], None
    arities = []
    for item in parts:
        if item.kind == 'map':
            continue  # trailing declaration metadata map is not an expression
        if item.kind != 'list' or not item.children or item.children[0].kind != 'vector':
            return [], 'unsupported_arity_shape'
        arities.append((item.children[0], item.children[1:]))
    return (arities, None) if arities else ([], 'missing_arity')


def _fn_parameter_contract(args):
    """Compiler-shaped fn parameter bindings and recur arity.

    In a variadic fn, recur receives the *single* rest sequence binding as
    one argument, in addition to all fixed arguments. Destructured patterns
    count as one positional argument, not one per bound local.
    """
    if args.kind != 'vector':
        return None, 'function_parameters_not_a_vector'
    parts=list(args.children)
    ampersands=[i for i,x in enumerate(parts)
                if x.kind=='atom' and x.value=='&']
    if len(ampersands)>1 or (ampersands and ampersands[0]!=len(parts)-2):
        return None, 'function_variadic_parameter_shape_invalid'
    variadic=bool(ampersands)
    patterns=[x for x in parts if not (x.kind=='atom' and x.value=='&')]
    names=set()
    for item in patterns:
        bound=binding_names(item)
        if bound is None or names.intersection(bound):
            return None, 'function_parameter_binding_invalid_or_duplicate'
        names.update(bound)
    return {'parameter_binding_names':tuple(sorted(names)),
            'fixed_parameter_count':len(patterns)-(1 if variadic else 0),
            'variadic':variadic,
            'recur_argument_count':len(patterns)},None


@dataclass
class _Context:
    name: str
    phase: str
    max_nodes: int = MAX_FLOW_NODES
    core_short_circuit: dict | None = None

    def __post_init__(self):
        self.nodes = []
        self.edges = []
        self.reads = []
        self.writes = []
        self.defaults = []
        self.interop = []
        self.unknown = []
        self._counter = 0
        self._binding_counter = 0
        # A path-specific CFG may duplicate one source expression (finally).
        # Its lexical declaration and reads retain ONE physical source ID.
        self._bindings_by_source = {}
        self._read_sites = set()
        self._write_sites = set()
        self._interop_sites = set()
        self._default_sites = set()
        self._evaluation_phase = "immediate"
        # Stack of lexical try scopes; a frame tracks possible *explicit*
        # throws, not arbitrary implicit exceptions from calls/Java interop.
        # Each frame owns candidate catches for its BODY, never its CATCH.
        self._try_frames = []

    def node(self, kind, form, *, synthetic=False):
        if self._counter >= self.max_nodes:
            raise OverflowError('flow_node_limit')
        self._counter += 1
        node_id = f'{self.name}#flow{self._counter}'
        self.nodes.append({'id': node_id, 'kind': kind, 'location': _loc(form),
                           'synthetic': bool(synthetic), 'phase': self.phase})
        return node_id

    def connect(self, predecessors, target, label='next'):
        for prev in predecessors:
            self.edges.append({'source': prev, 'target': target, 'relation': label,
                               'proof': 'core_special_form_syntax_only'})

    def issue(self, form, reason):
        if len(self.unknown) < MAX_DIAGNOSTICS:
            self.unknown.append({'reason': reason, 'location': _loc(form)})

    def route_explicit_exception(self, source, form):
        """Candidate catch dispatch, plus unmatched propagation out of a try.

        Catch tests depend on compiler-resolved Java types. The same explicit
        throw may match a catch or escape, so keep both possible paths. While
        evaluating a catch, the same try's handlers must not catch it again.
        """
        if not self._try_frames:
            self.issue(form, 'exception_handler_target_not_resolved')
            return
        frame = self._try_frames[-1]
        if frame['region'] == 'body' and frame['handlers']:
            for target in frame['handlers']:
                self.connect([source], target, 'exception_handler_candidate')
            self.issue(form, 'exception_catch_type_match_unresolved')
        frame['exceptional'].append(source)

    def observe_source_once(self, destination, seen, record, *identity):
        """Keep one lexical observation per physical source even across CFG copies.

        CFG paths remain separate, but cloned exceptional/normal finally nodes
        are not additional *textual* reads/writes/defaults/interops.
        """
        location = record.get('location') or {}
        key = (*identity, location.get('line'), location.get('column'),
               location.get('end_line'), location.get('end_column'),
               record.get('evaluation_phase', self._evaluation_phase))
        if key not in seen:
            seen.add(key)
            destination.append(record)

    def make_binding(self, name, kind, form):
        key = (name, kind, form.line, form.col, form.end_line, form.end_col)
        if key not in self._bindings_by_source:
            self._binding_counter += 1
            self._bindings_by_source[key] = {
                'id': f'{self.name}#binding{self._binding_counter}',
                'name': name, 'kind': kind, 'location': _loc(form)}
        return self._bindings_by_source[key]

    def reads_in(self, form, env):
        """Lexical Var *reads* without implying evaluation of data/macro bodies."""
        if form.kind == 'opaque':
            self.issue(form, 'opaque_reader_form_not_semantically_traversed')
            return
        if form.kind == 'atom':
            symbol = form.value
            if symbol in env and _identifier(symbol):
                identity = env[symbol]
                key = (identity['id'], form.line, form.col, form.end_line,
                       form.end_col, self._evaluation_phase)
                if key not in self._read_sites:
                    self._read_sites.add(key)
                    self.reads.append({'binding': identity, 'symbol': symbol,
                                       'location': _loc(form), 'proof': 'lexical_scope_binding',
                                       'evaluation_phase':self._evaluation_phase})
            return
        if form.kind == 'list' and _head(form) in {'quote', 'clojure.core/quote',
                                                  'comment', 'clojure.core/comment'}:
            return
        if form.kind == 'list' and _head(form) in {'fn', 'fn*'}:
            # A fn body is not executed at closure creation. Its local symbol
            # references are nonetheless lexical evidence, distinguishable
            # from immediate value reads at runtime.
            forms=list(form.children[1:])
            if forms and forms[0].kind=='atom':
                forms=forms[1:]  # optional self-name, handled below by normal scope
            if forms and forms[0].kind=='vector':
                variants=[(forms[0],forms[1:])]
            elif forms and all(part.kind=='list' and part.children and
                               part.children[0].kind=='vector' for part in forms):
                variants=[(part.children[0],part.children[1:]) for part in forms]
            else:
                self.issue(form,'invalid_fn_arity_shape')
                return
            previous=self._evaluation_phase
            self._evaluation_phase='deferred_lambda'
            try:
                for params,body in variants:
                    plan=function_contract_plan(params,body,
                                                expand_contracts=_head(form)=='fn')
                    if plan is None:
                        self.issue(params,'nested_fn_contract_shape_unverified')
                        continue
                    pre,body_expressions,post=plan
                    local=dict(env)
                    valid=True
                    for pattern in params.children:
                        if pattern.kind=='atom' and pattern.value=='&':
                            continue
                        names=binding_names(pattern)
                        if names is None:
                            valid=False;break
                        # Read source defaults as deferred lexical expressions.
                        self.read_destructuring_defaults(pattern,local)
                        for name in sorted(names):
                            local[name]=self.make_binding(name,'lambda_parameter',pattern)
                    if not valid:
                        self.issue(params,'unsupported_fn_binding_pattern')
                        continue
                    for item in pre:self.reads_in(item,local)
                    for item in body_expressions:self.reads_in(item,local)
                    if post:
                        result_env=dict(local)
                        result_env['%']=self.make_binding('%','postcondition_result',
                              body_expressions[-1] if body_expressions else params)
                        for item in post:self.reads_in(item,result_env)
            finally:
                self._evaluation_phase=previous
            return
        if form.kind == 'list' and _head(form) == 'letfn':
            # letfn's function names are mutually recursive and visible in
            # all initializers and in the body. Do not treat their definitions
            # as ordinary evaluated forms.
            if len(form.children) < 3 or form.children[1].kind != 'vector':
                self.issue(form, 'invalid_letfn_shape');return
            definitions=form.children[1].children
            if any(item.kind != 'list' or len(item.children) < 3 or
                   item.children[0].kind != 'atom' or
                   not _identifier(item.children[0].value)
                   for item in definitions):
                self.issue(form, 'invalid_letfn_definition');return
            names=[item.children[0].value for item in definitions]
            if len(set(names)) != len(names):
                self.issue(form,'duplicate_letfn_binding');return
            local=dict(env)
            for item in definitions:
                name=item.children[0].value
                local[name]=self.make_binding(name,'letfn_local_function',item.children[0])
            for item in definitions:
                # Rewrite only the head marker, keeping original source
                # locations, to reuse the nested-fn lexical scope analyzer.
                from .clojure_forms import Form
                synthetic=Form('atom','fn*',(),item.line,item.col)
                fn=Form('list','(',tuple([synthetic,*item.children[1:]]),
                        item.line,item.col,item.end_line,item.end_col)
                self.reads_in(fn,local)
            for body in form.children[2:]:self.reads_in(body,local)
            return
        if form.kind == 'list' and _head(form) in {'syntax-quote', 'binding', 'with-open'}:
            self.issue(form, 'nested_macro_lexical_environment_unmodeled')
            return
        if form.kind == 'list' and _head(form) in {'let', 'let*', 'loop', 'loop*'}:
            self.let_bindings(form, env, collect_only=True)
            return
        if form.kind == 'list' and _head(form) == 'if':
            for item in form.children[1:]:
                self.reads_in(item, env)
            return
        if form.kind == 'list' and _head(form) == 'do':
            for item in form.children[1:]:
                self.reads_in(item, env)
            return
        if form.kind == 'list' and _head(form) == 'try':
            parts=list(form.children[1:])
            body=[];catches=[];finally_form=None;phase='body'
            for part in parts:
                kind=_head(part)
                if kind=='catch' and phase!='finally':
                    phase='catch';catches.append(part)
                elif kind=='finally' and phase!='finally':
                    phase='finally';finally_form=part
                elif phase=='body':body.append(part)
                else:self.issue(part,'invalid_try_clause_order');return
            for part in body:self.reads_in(part,env)
            for clause in catches:
                if (len(clause.children)<4 or clause.children[2].kind!='atom' or
                        not _identifier(clause.children[2].value)):
                    self.issue(clause,'invalid_catch_binding');continue
                catch_env=dict(env)
                symbol=clause.children[2].value
                catch_env[symbol]=self.make_binding(symbol,'catch_local',clause.children[2])
                for part in clause.children[3:]:self.reads_in(part,catch_env)
            if finally_form:
                for part in finally_form.children[1:]:self.reads_in(part,env)
            return
        if form.kind == 'list' and _head(form) in {'catch', 'finally'}:
            self.issue(form, 'catch_or_finally_outside_try')
            return
        if form.kind == 'list' and _head(form) == 'set!':
            # The assignment target is NOT a value read. Clojure locals are
            # immutable; even a same-spelled lexical binding is not a proven
            # writable Var. Reflection/field target identity remains unknown.
            if len(form.children) != 3:
                self.issue(form,'set_bang_invalid_arity');return
            target,value=form.children[1:]
            self.reads_in(value,env)
            if target.kind=='atom' and _identifier(target.value):
                self.observe_source_once(self.writes,self._write_sites,{'target_syntax':target.value,
                    'target_kind':'global_var_candidate', 'location':_loc(target),
                    'resolution':'verified_write_syntax_target_unresolved'}, 'global_var_candidate', target.value)
                if target.value in env:
                    self.issue(target,'set_bang_cannot_assign_immutable_local')
            elif target.kind=='list' and _head(target)=='.' and len(target.children)>=3:
                self.reads_in(target.children[1],env)
                self.observe_source_once(self.writes,self._write_sites,{'target_syntax':'.',
                    'target_kind':'java_or_js_field_candidate','location':_loc(target),
                    'resolution':'verified_write_syntax_target_unresolved'}, 'java_or_js_field_candidate', '.')
            else:
                self.issue(target,'set_bang_target_unresolved')
            return
        if form.kind == 'list' and _head(form) in {'recur', 'throw'}:
            for item in form.children[1:]:
                self.reads_in(item, env)
            return
        if form.kind == 'list' and _head(form) in {'def', 'defn', 'defmacro'}:
            self.issue(form, 'nested_global_definition_not_resolved')
            return
        if form.kind == 'list' and _head(form) in {'var', 'clojure.core/var'}:
            return  # Var-quote refers to a Var, never a local binding
        # A short-circuit form nested inside an otherwise opaque function
        # application has no proven containing CFG. Still, its later lexical
        # reads must NOT be labeled unconditional. Do not fabricate a CFG
        # transfer across an arbitrary possible macro/operator.
        nested_macro = (self.core_short_circuit or {}).get(_head(form))
        if form.kind == 'list' and nested_macro in ('and', 'or') and _head(form) not in env:
            self.issue(form, 'nested_core_short_circuit_cfg_unavailable')
            for index, operand in enumerate(form.children[1:]):
                previous_phase = self._evaluation_phase
                if index:
                    self._evaluation_phase = 'short_circuit_conditional'
                try:
                    self.reads_in(operand, env)
                finally:
                    self._evaluation_phase = previous_phase
            return
        if form.kind == 'list' and _head(form) in {'if-let', 'when-let', 'when', 'for',
                                                   'doseq', 'as->', 'cond', 'case'}:
            self.issue(form, 'macro_expansion_required_for_lexical_reads')
            return
        if form.kind == 'list':
            h = _head(form)
            if h == '.' or h.startswith('.-') or (h.startswith('.') and len(h)>1):
                self.observe_source_once(self.interop,self._interop_sites,
                    {'expression': h, 'location': _loc(form),
                     'resolution': 'unresolved_java_or_js_interop'}, h)
            elif h and '/' in h and not h.startswith(':'):
                prefix = h.split('/', 1)[0]
                if '.' in prefix or prefix[:1].isupper():
                    self.observe_source_once(self.interop,self._interop_sites,
                        {'expression': h, 'location': _loc(form),
                         'resolution': 'unresolved_static_member_or_qualified_var'}, h)
        for child in form.children:
            self.reads_in(child, env)

    def destructuring_defaults(self, pattern, predecessors, env, *, label='next',
                               recur_target=None):
        """Conservative flow for literal map :or expressions, including nesting.

        The Clojure 1.12 compiler can eagerly evaluate default expressions
        even if their input keys are present. Their ordering and the complete
        generated binding environment are not independently compiler-verified
        here. Read known *earlier* bindings only, never assume a name in the
        current pattern denotes an outer Var or a completed new local.
        """
        if not predecessors or not _has_map_default_values(pattern):
            return predecessors, []
        defaults = _default_expression_forms(pattern)
        names = binding_names(pattern)
        if defaults is None or names is None:
            self.issue(pattern, 'destructuring_default_shape_unverified')
            return predecessors, []
        # A symbol also bound in this pattern may refer to a prior outer
        # binding or one generated by destructure. Do not fabricate identity.
        safe_env = {name: identity for name, identity in env.items()
                    if name not in names}
        if names.intersection(env):
            self.issue(pattern, 'destructuring_default_binding_identity_conservative')
        current = predecessors
        evidence = []
        for index, default in enumerate(defaults):
            if not current:
                break  # A preceding default terminated the core-form path.
            step = self.node('destructuring_default_evaluation_candidate',
                             default, synthetic=True)
            self.connect(current, step, label if index == 0 else 'next_default')
            current = self.expr(default, [step], safe_env,
                                label='default_expression_candidate',
                                tail=False, recur_target=recur_target)
            evidence.append({'location': _loc(default),
                             'proof': 'literal_map_or_source_expression',
                             'may_evaluate_when_input_key_present': True,
                             'runtime_execution_confirmed': False,
                             'binding_order_compiler_verified': False,
                             'cfg_node_emitted': True,
                             'evaluation_phase': self._evaluation_phase})
        for item in evidence:
            self.observe_source_once(self.defaults,self._default_sites,item,
                                     item['proof'],item['cfg_node_emitted'])
        return current, evidence

    def condition_checks(self, expressions, predecessors, env, role, *,
                         recur_target=None):
        """Branch on unknown compile-time *assert*, preserving result value.

        A false enabled assertion may throw. The alternative skip edge means
        that *assert* may have been disabled when the function was compiled;
        no branch asserts a runtime observation or compiler-backed expansion.
        """
        current = predecessors
        for condition in expressions:
            if not current:
                return []
            gate = self.node(role + '_assertions_gate', condition, synthetic=True)
            self.connect(current, gate, role + '_candidate')
            evaluated = self.expr(condition, [gate], env,
                                  label='assertions_enabled_candidate',
                                  tail=False, recur_target=recur_target)
            check = self.node(role + '_test', condition, synthetic=True)
            self.connect(evaluated, check, 'test_result_candidate')
            failure = self.node(role + '_assertion_failure_candidate',
                                condition, synthetic=True)
            self.connect([check], failure,
                         'false_assertion_error_candidate')
            continuation = self.node(role + '_continue_candidate',
                                     condition, synthetic=True)
            self.connect([check], continuation, 'true_candidate')
            self.connect([gate], continuation,
                         'assertions_disabled_candidate')
            current = [continuation]
        return current

    def read_destructuring_defaults(self, pattern, env):
        """Account for :or source reads inside expressions without a CFG plan."""
        if not _has_map_default_values(pattern):
            return
        names=binding_names(pattern)
        defaults=_default_expression_forms(pattern)
        if names is None or defaults is None:
            self.issue(pattern,'destructuring_default_shape_unverified')
            return
        safe_env={k:v for k,v in env.items() if k not in names}
        if names.intersection(env):
            self.issue(pattern,'destructuring_default_binding_identity_conservative')
        for default in defaults:
            item = {'location':_loc(default),
                'proof':'literal_map_or_source_expression',
                'may_evaluate_when_input_key_present':True,
                'runtime_execution_confirmed':False,
                'binding_order_compiler_verified':False,
                'cfg_node_emitted':False,
                'evaluation_phase':self._evaluation_phase}
            self.observe_source_once(self.defaults,self._default_sites,item,
                                     item['proof'],item['cfg_node_emitted'])
            self.reads_in(default,safe_env)

    def let_bindings(self, form, env, *, collect_only=False):
        if len(form.children) < 3 or form.children[1].kind != 'vector':
            self.issue(form, 'invalid_let_binding_vector')
            return None
        items = form.children[1].children
        if len(items) % 2:
            self.issue(form, 'odd_let_binding_vector')
            return None
        local = dict(env)
        for i in range(0, len(items), 2):
            pattern, value = items[i], items[i+1]
            self.reads_in(value, local)
            names = binding_names(pattern)
            if names is None:
                self.issue(pattern, 'unmodeled_destructuring_binding')
                return None
            self.read_destructuring_defaults(pattern,local)
            for name in sorted(names):
                local[name] = self.make_binding(name, 'local', pattern)
        if collect_only:
            for body in form.children[2:]:
                self.reads_in(body, local)
        return local

    def expr(self, form, predecessors, env, *, label='next', tail=False, recur_target=None):
        """Conservative core-special-form CFG; no user-defined macro is expanded.

        `recur` creates a backedge only for a syntactically supported tail
        position with matching arity. A non-tail `recur` is invalid Clojure;
        it is diagnosed without claiming the edge.
        """
        # Once a known special form terminates a path (throw/recur), later
        # expressions in a do/let branch must not resurrect normal flow.
        # An exception handler is entered separately with an explicit node.
        if not predecessors:
            return []
        h = _head(form)
        if form.kind == 'opaque':
            self.issue(form, 'opaque_reader_form_cfg_unresolved')
            step = self.node('opaque_expression', form)
            self.connect(predecessors, step, label)
            return [step]
        macro = (self.core_short_circuit or {}).get(h)
        if (form.kind == 'list' and macro in ('and', 'or') and h not in env):
            # The short-circuit expansion uses a conditional nested `if`.
            # Branches are possible source-level paths, NOT proof that a
            # particular predicate is truthy or that the macro expanded.
            # `recur` may occur in the last operand's tail position.
            start = self.node('core_short_circuit_candidate', form, synthetic=True)
            self.connect(predecessors, start, label)
            operands = form.children[1:]
            if not operands:
                literal = self.node('core_and_true_candidate' if macro == 'and'
                                    else 'core_or_nil_candidate', form, synthetic=True)
                self.connect([start], literal, 'empty_core_macro_candidate')
                return [literal]
            current = [start]
            exits = []
            for i, operand in enumerate(operands):
                previous_phase = self._evaluation_phase
                if i:
                    self._evaluation_phase = 'short_circuit_conditional'
                try:
                    current = self.expr(operand, current, env,
                                        label='core_macro_operand_candidate',
                                        tail=tail and i == len(operands)-1,
                                        recur_target=recur_target)
                finally:
                    self._evaluation_phase = previous_phase
                if not current:
                    break  # A proven explicit throw/recur cannot fall through.
                if i < len(operands)-1:
                    stop = self.node('short_circuit_stop_candidate', operand, synthetic=True)
                    self.connect(current, stop, 'short_circuit_stop_candidate')
                    exits.append(stop)
                    continuation = self.node('short_circuit_continue_candidate',
                                             operand, synthetic=True)
                    self.connect(current, continuation, 'short_circuit_continue_candidate')
                    current = [continuation]
                else:
                    exits.extend(current)
            self.issue(form, 'core_short_circuit_source_only_not_macroexpanded')
            return exits
        if h == 'if' and len(form.children) in {3, 4}:
            test = form.children[1]
            # `if` must first evaluate its test, including nested *core*
            # special forms. Merely scanning lexical reads made an explicit
            # (throw ...) test look as though it returned normally, creating
            # invented branch/postcondition return paths.
            test_exits = self.expr(test, predecessors, env, label=label,
                                   tail=False, recur_target=recur_target)
            if not test_exits:
                return []
            start = self.node('if_test', test)
            self.connect(test_exits, start, 'test_value_candidate')
            exits = self.expr(form.children[2], [start], env, label='true',
                              tail=tail, recur_target=recur_target)
            if len(form.children) == 4:
                exits += self.expr(form.children[3], [start], env, label='false',
                                   tail=tail, recur_target=recur_target)
            else:
                nil_node = self.node('implicit_nil', form, synthetic=True)
                self.connect([start], nil_node, 'false')
                exits += [nil_node]
            return exits
        if h == 'do':
            current = predecessors
            body = form.children[1:]
            for i, item in enumerate(body):
                current = self.expr(item, current, env,
                                    label=label if i == 0 else 'next',
                                    tail=tail and i == len(body)-1,
                                    recur_target=recur_target)
            return current
        if h in {'let', 'let*', 'loop', 'loop*'} and len(form.children) >= 3:
            binds = form.children[1]
            if binds.kind == 'vector' and len(binds.children) % 2 == 0:
                current = predecessors
                local = dict(env)
                for i in range(0, len(binds.children), 2):
                    pattern, value = binds.children[i:i+2]
                    current = self.expr(value, current, local,
                                        label=label if i == 0 else 'next',
                                        tail=False, recur_target=recur_target)
                    if not current:
                        return []  # Do not resurrect an unreachable loop entry.
                    names = binding_names(pattern)
                    if names is None:
                        self.issue(pattern, 'unmodeled_destructuring_binding')
                        break
                    current, _ = self.destructuring_defaults(
                        pattern, current, local, label='destructure_defaults',
                        recur_target=recur_target)
                    if not current:
                        return []
                    for name in sorted(names):
                        local[name] = self.make_binding(name, 'local', pattern)
                else:
                    target = recur_target
                    if h in {'loop', 'loop*'}:
                        header = self.node('loop_entry', binds, synthetic=True)
                        self.connect(current, header, 'loop_enter')
                        current = [header]
                        target = (header, len(binds.children)//2, len(self._try_frames))
                    body = form.children[2:]
                    for i, part in enumerate(body):
                        current = self.expr(part, current, local,
                                            tail=i == len(body)-1 and (tail or h in {'loop', 'loop*'}),
                                            recur_target=target)
                    return current
            self.issue(form, 'let_or_loop_cfg_not_proven')
        if h == 'try':
            # A typed catch is a candidate, never an exact match without
            # verified compiler type information. But the *ordering* of
            # explicit throws, catch bodies, and finally is a core rule.
            parts = list(form.children[1:])
            body = []; catches = []; final_form = None; state = 'body'
            invalid = False
            for part in parts:
                kind = _head(part)
                if kind == 'catch' and state != 'finally':
                    state = 'catch'; catches.append(part)
                elif kind == 'finally' and state != 'finally':
                    state = 'finally'; final_form = part
                elif state == 'body':
                    body.append(part)
                else:
                    invalid = True
            if (invalid or
                    any(len(c.children) < 3 or c.children[1].kind != 'atom' or
                        not _identifier(c.children[1].value) or
                        c.children[2].kind != 'atom' or
                        not _identifier(c.children[2].value) for c in catches)):
                self.issue(form, 'try_cfg_invalid_clause_shape')
                step = self.node('opaque_try', form)
                self.connect(predecessors, step, label)
                return []  # Malformed try: no invented normal return.
            start = self.node('try_entry', form, synthetic=True)
            self.connect(predecessors, start, label)
            catch_nodes = [self.node('catch_candidate', c) for c in catches]
            frame = {'handlers': catch_nodes, 'exceptional': [], 'region': 'body'}
            self._try_frames.append(frame)
            try:
                current = [start]
                for i, part in enumerate(body):
                    current = self.expr(part, current, env,
                                        label='try_body' if i == 0 else 'next',
                                        tail=tail and i == len(body)-1 and not final_form,
                                        recur_target=recur_target)
                if not body:
                    empty = self.node('try_implicit_nil', form, synthetic=True)
                    self.connect(current, empty, 'try_empty_body_nil')
                    current = [empty]
                exits = current[:]
                frame['region'] = 'catch'
                # Catch paths are only candidates, anchored to a specific
                # exception edge; absent an explicit throw, runtime calls
                # may still throw (we do not claim complete reachability).
                for clause, header in zip(catches, catch_nodes):
                    catch_env = dict(env)
                    symbol = clause.children[2].value
                    catch_env[symbol] = self.make_binding(symbol, 'catch_local',
                                                          clause.children[2])
                    branch = [header]
                    for i, part in enumerate(clause.children[3:]):
                        branch = self.expr(part, branch, catch_env,
                                           label='catch_body' if i == 0 else 'next',
                                           tail=tail and i == len(clause.children[3:])-1
                                                and not final_form,
                                           recur_target=recur_target)
                    if len(clause.children) == 3:
                        # catch class symbol alone is valid: body defaults to nil.
                        empty = self.node('catch_implicit_nil', clause, synthetic=True)
                        self.connect(branch, empty, 'catch_empty_body_nil')
                        branch = [empty]
                    exits.extend(branch)
            finally:
                self._try_frames.pop()

            pending = frame['exceptional']
            if final_form is not None:
                # Two source-identical but control-distinct executions:
                # normal completion and exceptional completion. Reusing a
                # single node would spuriously turn a throw into a return.
                # No finalizer is entered when the corresponding path is empty.
                def finalizer_path(inputs, kind, relation):
                    if not inputs:
                        return []
                    entry = self.node(kind, final_form, synthetic=True)
                    self.connect(inputs, entry, relation)
                    current = [entry]
                    for i, part in enumerate(final_form.children[1:]):
                        current = self.expr(part, current, env,
                                            label='finally_body' if i == 0 else 'next',
                                            tail=False, recur_target=None)
                    return current

                exits = finalizer_path(exits, 'finally_normal_entry', 'finally_normal')
                exceptional = finalizer_path(pending, 'finally_exceptional_entry',
                                             'finally_exceptional')
                if exceptional:
                    # The original exception continues ONLY if the finalizer
                    # itself completed normally. A throwing finalizer wins.
                    resumed = self.node('exception_rethrow_candidate', final_form,
                                        synthetic=True)
                    self.connect(exceptional, resumed, 'finally_preserves_exception_candidate')
                    self.route_explicit_exception(resumed, final_form)
            else:
                for candidate in pending:
                    self.route_explicit_exception(candidate, form)
            if final_form or pending:
                self.issue(form, 'implicit_exception_paths_and_catch_types_unproven')
            return exits
        if h in {'recur', 'throw'}:
            # JVM/Clojure evaluates operands before a throw or recur transfer.
            # Explicit terminating core forms in an operand suppress that
            # transfer; lexical `reads_in` alone cannot represent this.
            operands = form.children[1:]
            if h == 'throw' and len(operands) != 1:
                self.issue(form, 'throw_invalid_operand_count')
                invalid = self.node('invalid_throw', form)
                self.connect(predecessors, invalid, label)
                return []
            current = predecessors
            for i, operand in enumerate(operands):
                current = self.expr(operand, current, env,
                                    label=label if i == 0 else 'next_operand',
                                    tail=False, recur_target=recur_target)
                if not current:
                    return []
            step = self.node(h, form)
            self.connect(current, step, 'operands_evaluated_candidate'
                         if operands else label)
            if h == 'recur':
                if recur_target is None:
                    self.issue(form, 'recur_target_unavailable')
                elif (len(recur_target) >= 3 and
                      recur_target[2] != len(self._try_frames)):
                    self.issue(form, 'recur_cannot_cross_try')
                elif not tail:
                    self.issue(form, 'recur_not_in_tail_position')
                elif len(operands) != recur_target[1]:
                    self.issue(form, 'recur_arity_mismatch')
                else:
                    self.connect([step], recur_target[0], 'recur_backedge')
            else:
                self.route_explicit_exception(step, form)
            return []
        step = self.node('expression', form)
        self.connect(predecessors, step, label)
        self.reads_in(form, env)
        if h in {'catch','finally'}:
            self.issue(form,'exception_flow_cfg_unresolved')
        return [step]


def analyze_declaration(definition, *, namespace='', phase='runtime', max_nodes=MAX_FLOW_NODES,
                        core_short_circuit=None, implicit_field_forms=()):
    """Per-arity read-only lexical dataflow and core-special-form CFG.

    Returns a structured coverage record. Unmodeled constructs are explicitly
    marked; never interprets a successful parse as complete semantics.
    """
    arities, issue = _arity_body(definition)
    name = definition.children[1].value
    result={'status':'bounded_core_special_forms_only','declaration':
            namespace+'/'+name if namespace else name, 'arities':[], 'limitations':[]}
    if issue:
        result['limitations'].append(issue)
        return result
    for index, (args, body) in enumerate(arities):
        ctx=_Context(f'{result["declaration"]}@{definition.line}:{definition.col}/arity{index}',
                     phase, max_nodes, core_short_circuit)
        contract,contract_issue=_fn_parameter_contract(args)
        if contract is None:
            ctx.issue(args,contract_issue)
            result['arities'].append({'parameter_binding_proven':False,
                                     'unknown':ctx.unknown,'nodes':[],'edges':[],
                                     'reads':[],'writes':[],'interop':[]})
            continue
        names=contract['parameter_binding_names']
        env={}
        # deftype/defrecord fields are lexically visible in method bodies.
        # Their source identity is explicit, but their runtime object type and
        # mutability are NOT inferred from the binding.
        for field in implicit_field_forms:
            if field.kind=='atom':
                env[field.value]=ctx.make_binding(field.value,'type_field_source',field)
        contract_evidence={'status':'no_literal_function_conditions',
            'compiler_macroexpansion_confirmed':False,
            'runtime_assertions_enabled_unverified':True,
            'preconditions':[], 'postconditions':[],
            'implicit_result_binding':None}
        try:
            start=ctx.node('entry',args,synthetic=True)
            current=[start]
            # Every recur in this arity targets the beginning of the arity;
            # parameter destructuring can therefore be revisited.
            target=(start,contract['recur_argument_count'], 0)
            for pattern in args.children:
                if pattern.kind=='atom' and pattern.value=='&':
                    continue
                current, found = ctx.destructuring_defaults(
                    pattern, current, env, label='parameter_destructure',
                    recur_target=target)
                # Each emission was recorded by the bounded flow context.
                bound = binding_names(pattern)
                for symbol in sorted(bound):
                    env[symbol]=ctx.make_binding(symbol,'parameter',pattern)
            plan = function_contract_plan(args, body)
            if plan is None:
                ctx.issue(args,'function_contract_shape_unverified')
                opaque=ctx.node('opaque_function_contract',args,synthetic=True)
                ctx.connect(current,opaque,'contract_unresolved')
            else:
                pre,body_expressions,post=plan
                if pre or post:
                    contract_evidence['status']='conditional_literal_function_contract_candidate'
                    contract_evidence['preconditions']=[_loc(f) for f in pre]
                    contract_evidence['postconditions']=[_loc(f) for f in post]
                    ctx.issue(args,'assertions_compile_time_setting_unverified')
                current=ctx.condition_checks(pre,current,env,'precondition',
                                            recur_target=target)
                for i,part in enumerate(body_expressions):
                    current=ctx.expr(part,current,env,
                        tail=(i==len(body_expressions)-1),recur_target=target)
                if post and not current:
                    ctx.issue(args,'postconditions_no_normal_body_exit_in_bounded_cfg')
                if post and current:
                    # The postcondition's implicit % denotes the returned
                    # *value*, not a parameter, global Var or earlier local.
                    capture=ctx.node('function_result_capture_candidate',
                                     body_expressions[-1] if body_expressions else args,
                                     synthetic=True)
                    ctx.connect(current,capture,'body_result_candidate')
                    result_binding=ctx.make_binding('%','postcondition_result',
                        body_expressions[-1] if body_expressions else args)
                    contract_evidence['implicit_result_binding']=result_binding
                    post_env=dict(env)
                    post_env['%']=result_binding
                    current=ctx.condition_checks(post,[capture],post_env,
                        'postcondition',recur_target=target)
                end=ctx.node('normal_return',definition,synthetic=True)
                ctx.connect(current,end,'return')
        except OverflowError:
            ctx.issue(definition,'flow_node_limit')
            ctx.issue(definition,'flow_partial_due_to_limit')
        from .clojure_type_hints import literal_type_tag
        typed_parameters=[]
        for position,parameter in enumerate(args.children):
            if (tag:=literal_type_tag(parameter)) is not None:
                typed_parameters.append({'parameter_position':position,
                    'parameter':parameter.value if parameter.kind=='atom' else None,
                    'type_tag':tag})
        result['arities'].append({'parameter_binding_proven':True,
            'literal_parameter_type_hints':typed_parameters,
            'contract_evidence':contract_evidence,
            'destructuring_default_evidence':ctx.defaults,
            'parameters':list(names),
            'implicit_field_source_bindings':[
                env[field.value] for field in implicit_field_forms
                if field.kind=='atom' and field.value in env and
                env[field.value]['kind']=='type_field_source'],
            'declared_fixed_parameter_count':contract['fixed_parameter_count'],
            'declared_variadic':contract['variadic'],
            'recur_expected_argument_count':contract['recur_argument_count'],
            'nodes':ctx.nodes,'edges':ctx.edges,
            'reads':ctx.reads,'writes':ctx.writes,'def_use_edges':[{'source':r['binding'].get('id'),'target_location':r['location'], 'symbol':r['symbol'], 'relation':'lexical_def_use', 'proof':'lexical_scope_binding'} for r in ctx.reads], 'interop':ctx.interop,'unknown':ctx.unknown})
    return result
