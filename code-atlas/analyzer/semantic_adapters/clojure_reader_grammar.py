"""Conservative *reader validity* evidence, separate from source syntax retention.

The source reader deliberately preserves opaque tagged forms for lossless CST.
That is not evidence that the official reader accepts those forms. This module
never executes a project namespace, tagged reader, macro or data reader; it
returns `context_required` whenever reader settings or a compiler-specific
reader are necessary to make a decision.

`accepted_subset` is *not* a claim of complete grammar conformance: only the
forms and syntax checks explicitly implemented in the bounded source reader
are covered. This is a static, optional semantic gate for project analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from uuid import UUID
from typing import Mapping


_SCALAR_BUILTINS = frozenset(('#inst', '#uuid'))
_RECOGNIZED_READER_PREFIXES = frozenset(("'", '`', '#\'', '~', '~@', '@', '#('))
_UUID_CANONICAL = re.compile(r'(?i)[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z')
_ISO_DATE = re.compile(r'\d{4}-\d{2}-\d{2}\Z')
_SIMPLE_INSTANT = re.compile(
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?'
    r'(?:Z|[+-]\d{2}:\d{2})?\Z')


def _date_tag_is_valid(raw: str) -> bool | None:
    # We can certify only a strict, well-understood ISO-8601 subset here.
    # Reader-specific permissive forms remain context_required, NOT rejected.
    if _ISO_DATE.fullmatch(raw):
        try:
            date.fromisoformat(raw)
        except ValueError:
            return False
        return True
    if _SIMPLE_INSTANT.fullmatch(raw):
        try:
            datetime.fromisoformat(raw.replace('Z','+00:00'))
        except ValueError:
            return False
        return True
    # Clearly non-ISO strings cannot be accepted by the built-in reader.
    if not re.match(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}', raw):
        return False
    return None


def classify_source(source: str, *, platform: str,
                    context: str = 'project_source',
                    namespace: str | None = None,
                    aliases: Mapping[str,str] | None = None) -> dict:
    """Return bounded official-reader acceptance evidence for inert source.

    `platform='cljc'` yields separate `clj` and `cljs` status records. A plain
    `fresh_reader` represents the pinned reference reader with no custom tags
    or namespace aliases. `project_source` is conservative: a project may have
    reader context unavailable to a source-only inspection.
    """
    if platform not in ('clj','cljs','cljc'):
        raise ValueError('unsupported_source_platform')
    if context not in ('fresh_reader','project_source','compiler_builtins'):
        raise ValueError('unsupported_reader_context')
    if context == 'compiler_builtins' and platform != 'cljs':
        raise ValueError('compiler_builtins_only_for_cljs_profile')
    if namespace is not None and (not isinstance(namespace,str) or not namespace):
        raise ValueError('invalid_namespace_context')
    from .clojure_forms import lex, _Parser, ReaderError
    from .clojure_conditionals import project, ProjectionUnavailable

    parse_mode = 'shared' if platform == 'cljc' else platform
    try:
        parser = _Parser(lex(source,platform=parse_mode),platform=parse_mode)
        forms = parser.all()
        discarded_forms = parser.discarded_forms
    except (ReaderError,ValueError) as exc:
        # A .cljc source without reader conditionals can nevertheless have
        # different lexical validity in JVM vs vendored CLJS tools.reader.
        # E.g. `#_ \ :after` is invalid on JVM and valid on CLJS. Preserve
        # the two independently proved results rather than letting the JVM's
        # shared lexer silently reject the CLJS source projection.
        if platform == 'cljc' and '#?' not in source:
            variants={dialect:classify_source(source,platform=dialect,
                         context=context,namespace=namespace,aliases=aliases)
                      for dialect in ('clj','cljs')}
            outcomes=[record['status'] for record in variants.values()]
            overall=('rejected' if 'rejected' in outcomes else
                     'context_required' if 'context_required' in outcomes
                     else 'accepted_subset')
            return {'status':overall,'platforms':variants,
                    'context_policy':context,'reader_execution':False,
                    'complete_official_grammar_certified':False,
                    'dialect_specific_lexer_fallback':True}
        return {'status':'rejected', 'reason':'structural_reader_rejected',
                'detail_type':type(exc).__name__, 'platform':platform,
                'context_policy':context}

    def grade(selected, dialect: str) -> dict:
        failures = []
        uncertainties = []
        known_aliases = dict(aliases or {})

        def walk(form, fn_literal_depth: int = 0):
            if form.kind == 'opaque' and form.value == '#(':
                if fn_literal_depth:
                    failures.append('nested_reader_function_literal')
                fn_literal_depth += 1
            if form.kind == 'namespaced_map':
                value = form.value
                if value.startswith('#::'):
                    alias = value[3:]
                    if alias and alias not in known_aliases:
                        (failures if context != 'project_source' else uncertainties).append(
                            'namespaced_map_alias_requires_reader_context')
                    elif not alias and namespace is None and context == 'project_source':
                        uncertainties.append('current_namespace_required_for_auto_map')
            elif form.kind == 'regex':
                from .clojure_regex_syntax import definitely_invalid_regex
                invalid=definitely_invalid_regex(form.value)
                if invalid is not None:
                    failures.append(invalid)
                else:
                    from .clojure_regex_syntax import regex_has_version_dependent_property
                    if regex_has_version_dependent_property(form.value):
                        uncertainties.append('regex_unicode_alias_requires_jdk_profile')
            elif form.kind == 'atom':
                from .clojure_reader_lexemes import validate_atom
                invalid = validate_atom(form.value,platform=dialect)
                if invalid:
                    failures.append(invalid)
                if not form.value.startswith('::'):
                    pass
                else:
                    raw = form.value[2:]
                    if '/' in raw:
                        alias = raw.split('/',1)[0]
                        if alias not in known_aliases:
                            (failures if context == 'fresh_reader' else uncertainties).append(
                                'auto_keyword_alias_requires_reader_context')
                    elif not raw or (namespace is None and context == 'project_source'):
                        uncertainties.append('current_namespace_required_for_auto_keyword')
            elif form.kind == 'opaque':
                tag = form.value
                if tag in ('#?', '#?@'):
                    if platform != 'cljc':
                        failures.append('reader_conditionals_not_allowed_in_native_file')
                elif tag in _SCALAR_BUILTINS:
                    if len(form.children) != 1 or form.children[0].kind != 'string':
                        failures.append('builtin_reader_tag_requires_string')
                    else:
                        v = form.children[0].value
                        if tag == '#uuid':
                            if not _UUID_CANONICAL.fullmatch(v):
                                failures.append('malformed_uuid_literal')
                        else:
                            is_valid = _date_tag_is_valid(v)
                            if is_valid is False:
                                failures.append('malformed_inst_literal')
                            elif is_valid is None:
                                uncertainties.append('inst_format_requires_exact_jvm_reader')
                elif tag == '#js':
                    # In the CLJS compiler, cljs.tagged-literals/read-js is
                    # defined only for vectors/maps. Keys must be strings or
                    # *unqualified* keywords. Check source shape without
                    # invoking any tagged-reader function or compiler.
                    if len(form.children) != 1 or form.children[0].kind not in ('vector','map'):
                        failures.append('cljs_js_literal_requires_map_or_vector')
                    elif form.children[0].kind == 'map':
                        for key in form.children[0].children[::2]:
                            if not (key.kind == 'string' or
                                    (key.kind == 'atom' and key.value.startswith(':') and
                                     not key.value.startswith('::') and '/' not in key.value)):
                                failures.append('cljs_js_literal_invalid_map_key')
                                break
                    # The isolated JVM / vendor tools.reader reference has no
                    # default compiler data-reader registry for #js. With
                    # compiler context, *valid shape* is still not proof of
                    # literal evaluation, so it remains context_required.
                    if dialect == 'clj' or context == 'fresh_reader':
                        failures.append('js_tag_not_registered_in_selected_reader')
                    elif context == 'project_source':
                        uncertainties.append('cljs_js_tag_requires_compiler_reader_context')
                    # compiler_builtins: shape checked above; no project data
                    # reader is loaded or run by the production analyzer.
                elif tag == '#queue' and context == 'compiler_builtins':
                    if len(form.children) != 1 or form.children[0].kind != 'vector':
                        failures.append('cljs_queue_literal_requires_vector')
                elif tag.startswith('#') and tag not in _RECOGNIZED_READER_PREFIXES:
                    if context != 'project_source':
                        failures.append('unknown_tag_no_registered_data_reader')
                    else:
                        uncertainties.append('custom_data_reader_not_executed')
            for ch in form.children:
                walk(ch, fn_literal_depth)
            for meta in form.metadata:
                walk(meta, fn_literal_depth)

        for f in selected:
            walk(f)
        # The reader still reads the operand of #_ and can reject malformed
        # tags/aliases there. These operands must NEVER enter symbol graphs.
        # In .cljc, a discard within a nonselected #? branch requires a
        # dialect-aware provenance tree; until proven, fail closed.
        def selected_discard(item):
            """Only discard operands in the dialect-selected #? branches.

            The physical Form tree already retains the enclosing collection's
            source range even when #_ removes an inner form from semantics.
            Resolve first-match reader features using that actual source span.
            """
            def within(child, parent):
                start=(child.line,child.col)
                end=(child.end_line,child.end_col)
                return ((parent.line,parent.col) <= start and
                        end <= (parent.end_line,parent.end_col))

            def visit(form):
                if form.kind=='opaque' and form.value in ('#?','#?@') and form.children:
                    branch_list=form.children[0]
                    if branch_list.kind=='list':
                        branches=branch_list.children
                        chosen=None
                        for i in range(0,len(branches)-1,2):
                            feature=branches[i].value
                            if feature in (':'+dialect,':default'):
                                chosen=i+1
                                break
                        for i in range(1,len(branches),2):
                            if within(item,branches[i]) and chosen!=i:
                                return False
                for child in form.children:
                    if visit(child) is False:
                        return False
                return True

            return all(visit(root) is not False for root in forms)

        for discarded in discarded_forms:
            if platform == 'cljc' and not selected_discard(discarded):
                continue
            if platform == 'cljc':
                if dialect=='cljs' and discarded.kind=='opaque' and discarded.value=='#?@':
                    # Pinned CLJS tools.reader rejects discarded root splice,
                    # even if it would select an empty branch. JVM 1.12
                    # accepts the discarded form in this exact context.
                    failures.append('cljs_discarded_root_splice_not_supported')
                    continue
                try:
                    # Clojure JVM 1.12 permits the discard reader `#_` to
                    # consume a top-level splicing conditional in .cljc.
                    # The splice is discarded: it is never source semantics.
                    # The pinned CLJS vendor reader rejects this combination.
                    # Validate the JVM payload in a synthetic *vector only*
                    # so its selected forms can be grammar-checked without
                    # accidentally publishing semantic graph nodes.
                    if dialect=='clj' and discarded.kind=='opaque' and discarded.value=='#?@':
                        from .clojure_forms import Form
                        wrapper=Form('vector','[',(discarded,),discarded.line,
                                     discarded.col,(),discarded.end_line,discarded.end_col)
                        p=project((wrapper,),dialect,preserve_unknown_tagged_syntax=True)
                        partial_forms=p.forms[0].children
                    else:
                        p=project((discarded,),dialect,
                                  preserve_unknown_tagged_syntax=True)
                        partial_forms=p.forms
                except ProjectionUnavailable as exc:
                    failures.append('invalid_discarded_platform_reader_form')
                    continue
                for item in partial_forms:
                    walk(item)
            else:
                walk(discarded)
        if failures:
            status='rejected'
        elif uncertainties:
            status='context_required'
        else:
            status='accepted_subset'
        return {'status':status,'reasons':list(dict.fromkeys(failures+uncertainties)),
                'platform':dialect,'context_policy':context,
                'reader_execution':False,'semantic_binding_certified':False}

    if platform == 'cljc':
        platforms={}
        for dialect in ('clj','cljs'):
            try:
                projected=project(forms,dialect,preserve_unknown_tagged_syntax=True)
            except ProjectionUnavailable as exc:
                platforms[dialect]={'status':'rejected',
                    'reasons':['unavailable_platform_projection:'+str(exc)[:100]],
                    'platform':dialect,'context_policy':context,
                    'reader_execution':False,'semantic_binding_certified':False}
                continue
            platforms[dialect]=grade(projected.forms,dialect)
        outcomes=[v['status'] for v in platforms.values()]
        status=('rejected' if 'rejected' in outcomes else
                'context_required' if 'context_required' in outcomes else 'accepted_subset')
        return {'status':status,'platforms':platforms,
                'context_policy':context,'reader_execution':False,
                'complete_official_grammar_certified':False}
    return {**grade(forms,platform), 'complete_official_grammar_certified':False}
