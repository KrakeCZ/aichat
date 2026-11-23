"""Bounded, offline Clojure/CLJS structural form reader and namespace candidates.

This is a deliberately PARTIAL reader, not clojure.core/read, clj-kondo, a JVM
compiler, macro expander or a symbol/type resolver. It reads balanced forms,
strings and comments without running project code. Stage40 offers explicitly
platform-scoped, CANDIDATE .cljc conditional projections; not runtime semantics,
verified symbols, tagged literals, syntax-quote or evaluated requires.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import hashlib
from pathlib import Path
from typing import NamedTuple

from analyzer.symbol_graph import _file_id, _node_id
from .clojure_scope import referred_call_sites, OPAQUE_BINDING_MACROS

# Every discovered source is attempted; per-file limits remain fail-closed.
MAX_BYTES = 750_000
MAX_TOKENS = 50_000
MAX_DEPTH = 80
MAX_FORMS = 25_000
MAX_SYNTAX_NODES = 4000
MAX_UNRESOLVED_EXAMPLES = 80


class ReaderError(ValueError):
    pass


class Tok(NamedTuple):
    kind: str
    value: str
    line: int
    col: int
    end_line: int | None = None
    end_col: int | None = None


# Clojure 1.12: characters are a single character, one of six named forms,
# four hex unicode digits, or up to three octal digits <= 0o377. Reject invalid
# forms before emitting structural claims. Do NOT evaluate any token.
_CHAR_NAMES = frozenset({'newline','space','tab','formfeed','backspace','return'})
_CHAR_HEX = re.compile(r'u[0-9a-fA-F]{4}\Z')
_CHAR_OCTAL = re.compile(r'o[0-7]{1,3}\Z')
_SYMBOLIC_VALUES = frozenset({'##Inf','##-Inf','##NaN'})


def _validate_character(raw: str, line: int, col: int) -> None:
    token = raw[1:]
    if len(token) == 1 or token in _CHAR_NAMES:
        return
    if _CHAR_HEX.fullmatch(token):
        number = int(token[1:],16)
        if not 0xD800 <= number <= 0xDFFF:
            return
    if _CHAR_OCTAL.fullmatch(token) and int(token[1:],8) <= 0o377:
        return
    raise ReaderError(f'invalid Clojure character literal at {line}:{col}')


@dataclass(frozen=True)
class Form:
    kind: str
    value: str
    children: tuple['Form', ...]
    line: int
    col: int
    # Reader metadata attached to this exact form, never executed. Ordinary
    # metadata is not a claim of type/runtime identity or symbol resolution.
    metadata: tuple['Form', ...] = ()
    # End is exclusive, counted in Unicode code points (not UTF-8 bytes).
    # None means a synthetic/derived form without an exact textual span.
    end_line: int | None = None
    end_col: int | None = None


def lex(text: str, *, platform: str = 'clj') -> list[Tok]:
    """Tokenize bounded Clojure forms; comments/strings are never interpreted as code."""
    if '\x00' in text:
        raise ReaderError('NUL in source')
    out = []
    i, line, col = 0, 1, 1
    n = len(text)
    def advance(fragment):
        nonlocal line, col
        count = fragment.count('\n')
        if count:
            line += count
            col = len(fragment.rsplit('\n', 1)[-1]) + 1
        else:
            col += len(fragment)
    def emit(kind, val, start, stop, ln, column):
        if len(out) >= MAX_TOKENS:
            raise ReaderError('token limit')
        advance(text[start:stop])
        out.append(Tok(kind, val, ln, column, line, col))
    while i < n:
        ch = text[i]
        if ch.isspace() or ch == ',':
            advance(ch); i += 1; continue
        if ch == ';' or text.startswith('#!', i):
            end = text.find('\n', i)
            if end == -1: end = n
            advance(text[i:end]); i = end; continue
        ln, column, start = line, col, i
        if text.startswith('#:', i):
            # Namespaced maps (#:ns{}, #::alias{}, #::{}) are not unknown
            # tagged readers. Keep the prefix as a separate physical token so
            # the CST can account for any whitespace/comments before '{'.
            j = i + (3 if text.startswith('#::', i) else 2)
            while j < n and text[j] not in ' \t\r\n,;()[]{}\"':
                j += 1
            prefix = text[i:j]
            from .clojure_namespaced_maps import prefix_valid
            if not prefix_valid(prefix):
                raise ReaderError(f'invalid namespaced map prefix at {ln}:{column}')
            # LispReader NamespaceMapReader permits whitespace up to the map
            # delimiter, but not comment/discard macros in that gap. Ordinary
            # lexing would hide comments, so detect this before discarding.
            k = j
            while k < n and (text[k].isspace() or text[k] == ','):
                k += 1
            if k < n and (text[k] == ';' or text.startswith('#!',k)):
                raise ReaderError(f'comment between namespaced map prefix and body at {ln}:{column}')
            i = j
            emit('namespaced_prefix',prefix,start,i,ln,column)
            continue
        if text.startswith('#_', i):
            i += 2; emit('discard','#_', start, i, ln, column); continue
        if text.startswith('#=', i):
            raise ReaderError('read-eval form intentionally unsupported')
        # Symbolic IEEE values are zero-argument literals. Treating ##Inf as a
        # dispatch prefix would consume the NEXT top-level form and cause a
        # missing declaration in the graph. Unknown ## forms fail closed.
        if text.startswith('##', i):
            j=i+2
            while j < n and text[j] not in ' \t\r\n,;()[]{}\"':
                j+=1
            value=text[i:j]
            if value not in _SYMBOLIC_VALUES:
                raise ReaderError(f'unknown symbolic reader value at {ln}:{column}')
            emit('symbolic_value',value,i,j,ln,column)
            i=j
            continue
        if text.startswith('#?@',i) or text.startswith('#?', i):
            i += 3 if text.startswith('#?@',i) else 2
            emit('opaque_prefix',text[start:i],start,i,ln,column); continue
        if text.startswith("#'",i):
            i += 2; emit('opaque_prefix',"#'",start,i,ln,column); continue
        if text.startswith('#{',i):
            i += 2; emit('open','#{',start,i,ln,column); continue
        if text.startswith('#(',i):
            i += 2; emit('opaque_open','#(',start,i,ln,column); continue
        if text.startswith('#"',i):
            # Regex literal is opaque and must not create symbols.
            i += 1
            ch = '"'
        if ch == '"':
            is_regex = text.startswith('#"', start)
            i += 1
            decoded=[]
            terminated=False
            while i < n:
                if text[i] == '\\':
                    if i+1 >= n:break
                    if not is_regex:
                        escape=text[i+1]
                        if escape in ('\\','"','n','t','r','b','f'):
                            decoded.append(text[i:i+2]);i+=2;continue
                        if escape == 'u':
                            digits=text[i+2:i+6]
                            if len(digits)!=4 or any(c not in '0123456789abcdefABCDEF' for c in digits):
                                raise ReaderError(f'invalid unicode string escape at {line}:{col}')
                            decoded.append(text[i:i+6]);i+=6;continue
                        # JVM LispReader 1.12.6 uses backslash + up to 3
                        # octal digits (not backslash-o). ClojureScript uses
                        # backslash-oNNN, so shared .cljc must stay conservative.
                        # The pinned ClojureScript vendor tools.reader
                        # accepts the same backslash + octal digits as JVM.
                        # Backslash-o is not a string escape in this profile.
                        if escape in '01234567':
                            j=i+1
                            while j<n and j<i+4 and text[j] in '01234567':j+=1
                            if int(text[i+1:j], 8)>255:
                                raise ReaderError(f'out-of-range JVM octal escape at {line}:{col}')
                            decoded.append(text[i:j]);i=j;continue
                        raise ReaderError(f'unknown string escape at {line}:{col}')
                    # Regex literals are NOT Clojure strings: their backslashes
                    # belong to the regex engine. We do not compile regexes.
                    decoded.append(text[i:i+2]);i+=2;continue
                if text[i] == '"':
                    i += 1; terminated=True; break
                decoded.append(text[i]); i += 1
            if not terminated: raise ReaderError(f'unterminated string at {ln}:{column}')
            emit('regex' if is_regex else 'string',''.join(decoded),start,i,ln,column)
            continue
        if ch in '([{':
            i += 1; emit('open',ch,start,i,ln,column); continue
        if ch in ')]}':
            i += 1; emit('close',ch,start,i,ln,column); continue
        if text.startswith('~@',i):
            i += 2;emit('opaque_prefix','~@',start,i,ln,column);continue
        if ch in "'`~@^":
            i += 1;emit('meta' if ch=='^' else 'opaque_prefix',ch,start,i,ln,column);continue
        if ch == '\\':
            # Clojure character literal: \\newline, \\), \\u1234 etc.
            i += 1
            if i >= n: raise ReaderError(f'incomplete character literal at {ln}:{column}')
            first = text[i]
            i += 1
            if first.isspace() and platform != 'cljs':
                # The JVM reader consumes the first character after backslash,
                # even when it is whitespace, then finishes the character
                # token at the *next* separator.  Thus `\ ` at EOF and
                # `\  :x` are valid, whereas `\ :x` is an invalid
                # character token rather than two independent source forms.
                while i < n and text[i] not in ' \t\n\r,;()[]{}\\':
                    i += 1
            elif first.isalnum() or (ord(first) < 32 and not first.isspace()):
                # The official reader consumes the entire character token,
                # even when it starts with a literal control character.
                # Splitting `\<ctrl>1` into two accepted forms was incorrect.
                while i < n and text[i] not in ' \t\n\r,;()[]{}': i += 1
            _validate_character(text[start:i],ln,column)
            emit('char',text[start:i],start,i,ln,column);continue
        if ch == '#':
            # Opaque unknown reader dispatch (tag, #inst, #uuid, etc.). Parser
            # must not mistake the tag for an ordinary function call.
            i += 1
            while i < n and text[i] not in ' \t\r\n,;()[]{}"': i += 1
            if i == start+1: raise ReaderError(f'unsupported reader dispatch at {ln}:{column}')
            emit('opaque_prefix',text[start:i],start,i,ln,column);continue
        i += 1
        while i < n and text[i] not in ' \t\r\n,;()[]{}"\'`~@^':
            # In the official reader, '#' is a legal noninitial symbol and
            # keyword character (including syntax-quote auto-gensym x#).
            # Only a token BEGINNING with '#' is reader dispatch. Splitting
            # foo#bar here used to reject valid source and lose its symbols.
            i += 1
        token=text[start:i]
        from .clojure_reader_lexemes import validate_atom
        atom_error = validate_atom(token,platform=platform)
        if atom_error is not None:
            raise ReaderError(f'{atom_error} at {ln}:{column}')
        emit('atom',token,start,i,ln,column)
    return out


_CLOSE = {'(':')','[':']','{':'}','#{':'}','#(':')'}


class _Parser:
    def __init__(self, tokens: list[Tok], *, platform: str = 'clj'):
        self.tokens=tokens
        self.platform=platform
        self.pos=0
        self.visited=0
        self._anonymous_fn_depth=0
        self.discarded_forms=[]  # source-only reader validity, never semantic graph nodes
    def form(self, depth=0) -> Form:
        if depth > MAX_DEPTH: raise ReaderError('form depth limit')
        if self.pos >= len(self.tokens): raise ReaderError('missing form')
        t=self.tokens[self.pos];self.pos+=1
        self.visited+=1
        if self.visited > MAX_FORMS:raise ReaderError('form node limit')
        if t.kind == 'close':raise ReaderError(f'unexpected delimiter at {t.line}:{t.col}')
        if t.kind == 'opaque_prefix' and t.value == '#?@' and depth == 0:
            # Reader conditional splicing is not a top-level form: it only
            # makes sense within a containing sequential/map collection.
            raise ReaderError(f'top-level reader conditional splicing at {t.line}:{t.col}')
        if t.kind == 'namespaced_prefix':
            # A source-only map reader: auto-resolved namespaces are retained
            # as unresolved until a project namespace and alias are proven.
            value = self.form(depth+1)
            if value.kind != 'map':
                raise ReaderError(f'namespaced map must contain a map at {t.line}:{t.col}')
            result = Form('namespaced_map',t.value,value.children,t.line,t.col,
                          (),value.end_line,value.end_col)
            if (t.value.startswith('#:') and not t.value.startswith('#::') and
                    not any(child.kind=='opaque' and child.value in ('#?','#?@')
                            for child in value.children)):
                from .clojure_namespaced_maps import normalize_map, NamespaceMapUnavailable
                try:
                    normalize_map(result)
                except NamespaceMapUnavailable as exc:
                    raise ReaderError(f'{exc} at {t.line}:{t.col}') from exc
            return result
        if t.kind == 'discard':
            # Consume, discard and parse the next form. Top-level and nested
            # parsing use .forms() to skip discarded forms completely.
            self.discarded_forms.append(self.form(depth+1))  # source-only; never a semantic node
            return self.form(depth+1)  # reader macros skip discards before returning a value
        if t.kind in ('opaque_prefix','meta'):
            if t.kind=='meta':
                # Preserve *data* metadata on the exact next form. In Stage42
                # it was discarded, hiding :private on defn/defmacro names.
                meta_form=self.form(depth+1)
                value=self.form(depth+1)
                # Clojure metadata requires a map, keyword, symbol or string
                # *tag*. Metadata can attach only to IObj values: symbols and
                # persistent collections, not strings/numbers/keywords/chars.
                # This is a reader constraint, not evidence of a runtime type.
                from .clojure_shape import _number, UnsupportedShape
                if (meta_form.kind not in ('map', 'string', 'atom', 'vector') and
                        not (self.platform == 'shared' and meta_form.kind == 'opaque' and
                             meta_form.value == '#?')):
                    raise ReaderError(f'invalid metadata tag at {t.line}:{t.col}')
                if meta_form.kind == 'atom':
                    m = meta_form.value
                    if m in ('nil', 'true', 'false') or (m[:1].isdigit() or
                            (m[:1] in '+-' and len(m)>1 and m[1].isdigit())):
                        raise ReaderError(f'invalid metadata tag at {t.line}:{t.col}')
                if (value.kind not in ('list', 'vector', 'map', 'namespaced_map', 'set', 'atom') and
                        not (value.kind=='opaque' and value.value in ('#(', "#'", "'", '`', '~', '~@', '@'))):
                    raise ReaderError(f'cannot attach metadata to literal at {t.line}:{t.col}')
                if value.kind == 'atom':
                    atom = value.value
                    if (atom.startswith(':') or atom in ('nil', 'true', 'false') or
                            atom[:1].isdigit() or
                            (atom[:1] in '+-' and len(atom)>1 and atom[1].isdigit())):
                        raise ReaderError(f'cannot attach metadata to literal at {t.line}:{t.col}')
                return Form(value.kind,value.value,value.children,value.line,value.col,
                            (meta_form, *value.metadata),value.end_line,value.end_col)
            child=self.form(depth+1)
            return Form('opaque',t.value,(child,),t.line,t.col,(),
                        child.end_line,child.end_col)
        if t.kind in ('open','opaque_open'):
            if t.kind == 'opaque_open' and t.value == '#(':
                if self._anonymous_fn_depth:
                    raise ReaderError(f'nested reader function literal at {t.line}:{t.col}')
                self._anonymous_fn_depth += 1
            close=_CLOSE[t.value]
            content=[]
            while self.pos<len(self.tokens) and self.tokens[self.pos].kind!='close':
                if self.tokens[self.pos].kind=='discard':
                    self.pos+=1;self.discarded_forms.append(self.form(depth+1));continue
                content.append(self.form(depth+1))
            if self.pos>=len(self.tokens) or self.tokens[self.pos].value != close:
                raise ReaderError(f'unbalanced delimiter at {t.line}:{t.col}')
            # A map literal needs key/value pairs. Reader conditionals may
            # splice/omit forms at read time; in those cases report unresolved
            # instead of inventing a parity proof. No project code evaluated.
            if (t.value == '{' and len(content) % 2 and
                    not any(child.kind == 'opaque' and child.value in ('#?','#?@')
                            for child in content)):
                raise ReaderError(f'odd number of map entries at {t.line}:{t.col}')
            # The official JVM reader rejects duplicate literal map keys and
            # set members. Check only independently canonicalizable *data*
            # forms; unknown reader dispatch/conditionals remain unresolved.
            # This guard must never manufacture semantic certainty for opaque
            # forms or execute source expressions.
            if t.value in ('{','#{') and not any(
                    child.kind == 'opaque' for child in content):
                from .clojure_shape import literal_identity
                members = content[::2] if t.value == '{' else content
                seen_literals = set()
                for member in members:
                    value_key = literal_identity(member)
                    if value_key is None:
                        continue  # unknown is NOT evidence of uniqueness
                    if value_key in seen_literals:
                        raise ReaderError(
                            f'duplicate literal {"map key" if t.value == "{" else "set member"} '
                            f'at {member.line}:{member.col}')
                    seen_literals.add(value_key)
            closer=self.tokens[self.pos]
            self.pos+=1
            kind=('opaque' if t.kind=='opaque_open' else
                  'list' if t.value=='(' else 'vector' if t.value=='[' else
                  'map' if t.value=='{' else 'set')
            if t.kind == 'opaque_open' and t.value == '#(':
                self._anonymous_fn_depth -= 1
            return Form(kind,t.value,tuple(content),t.line,t.col,(),
                        closer.end_line,closer.end_col)
        return Form(t.kind,t.value,(),t.line,t.col,(),t.end_line,t.end_col)
    def all(self):
        result=[]
        while self.pos < len(self.tokens):
            if self.tokens[self.pos].kind=='discard':
                self.pos+=1;self.discarded_forms.append(self.form(1));continue
            result.append(self.form())
        return result


def read_forms(text: str) -> list[Form]:
    return _Parser(lex(text)).all()


def _reader_limitations(tokens: list[Tok], forms: list[Form]) -> list[dict]:
    """Bounded locations for reader features the subset cannot resolve.

    Never interpret these as fully resolved data or executable calls.
    """
    labels = {
        '#?': 'reader_conditional_platform_not_resolved',
        '#?@': 'reader_conditional_splice_not_resolved',
        '#(': 'anonymous_function_literal_not_resolved',
        '`': 'syntax_quote_macro_expansion_not_resolved',
        "'": 'quoted_data_not_executable',
        "#'": 'var_quoted_data_not_executable',
        '^': 'metadata_not_semantically_resolved',
        '#_': 'discarded_form_ignored',
        '~': 'unquote_macro_context_not_resolved',
        '~@': 'unquote_splice_macro_context_not_resolved',
    }
    seen: dict[str, dict] = {}
    def note(category: str, t: Tok) -> None:
        item = seen.setdefault(category, {'reason': category, 'count': 0, 'examples': []})
        item['count'] += 1
        if len(item['examples']) < 3:
            item['examples'].append({'line': t.line, 'column': t.col})
    for t in tokens:
        if t.kind in ('opaque_prefix', 'opaque_open'):
            note(labels.get(t.value, 'unknown_reader_dispatch_not_resolved'), t)
        elif t.kind == 'discard':
            note(labels['#_'], t)
        elif t.kind == 'meta':
            note(labels['^'], t)
        elif t.kind == 'regex':
            note('regex_literal_data_not_executable', t)
    for f in forms:
        if _top(f, 'ns'):
            for clause in f.children[2:]:
                if (clause.kind == 'list' and clause.children and
                        _sym(clause.children[0]) in {':require', ':require-macros'}):
                    for lib in clause.children[1:]:
                        if lib.kind not in {'atom', 'vector'}:
                            note('complex_namespace_require_not_resolved',
                                 Tok('list', ':require', lib.line, lib.col))
        if _definition(f) and _sym(f.children[0]) in {'defmacro', 'defmulti'}:
            note('macro_or_multimethod_runtime_dispatch_not_resolved',
                 Tok('atom', _sym(f.children[0]), f.line, f.col))
    return [seen[k] for k in sorted(seen)]


def _sym(form: Form | None) -> str:
    return form.value if form is not None and form.kind == 'atom' else ''


def _top(form: Form, head: str) -> bool:
    return form.kind=='list' and len(form.children)>=2 and _sym(form.children[0])==head


def _libspec_issue(spec: Form, platform: str = 'clj') -> str | None:
    """Only accept libspec structures we can account for without executing ns.

    This is *not* full ns/require syntax support. Unknown options are an
    unresolved dependency, never authorization to publish a guessed edge.
    """
    from .clojure_libspecs import simple_symbol
    if spec.kind == 'atom':
        return None if simple_symbol(spec) else 'invalid_lib_symbol'
    if spec.kind != 'vector' or not spec.children:
        return 'unsupported_require_libspec_shape'
    target=_sym(spec.children[0])
    if not simple_symbol(spec.children[0]):
        return 'invalid_require_target'
    options=spec.children[1:]
    i=0
    seen=set()
    # Only the literal ClojureScript ns-sugar forms are accepted here. They
    # remain *candidate* compiler-phase dependencies, never resolved vars.
    allowed={':as',':as-alias',':refer',':rename'}
    if platform == 'cljs':
        allowed |= {':include-macros', ':refer-macros'}
    while i<len(options):
        key=_sym(options[i]);i+=1
        if key not in allowed or key in seen:
            return 'unknown_or_repeated_require_option'
        seen.add(key)
        if i>=len(options):return 'missing_require_option_value'
        value=options[i];i+=1
        if key in {':as',':as-alias'}:
            name=_sym(value)
            if not simple_symbol(value):
                return 'invalid_namespace_alias'
        elif key==':refer':
            if _sym(value)==':all':
                if platform != 'clj':
                    return 'refer_all_not_supported_or_ambiguous_on_cljs'
                continue  # JVM namespace load, but Vars not enumerated
            if value.kind not in ('vector','list') or not value.children:
                return 'unsupported_refer_shape'
            items=[_sym(x) for x in value.children]
            if any(not simple_symbol(x) for x in value.children):
                return 'invalid_refer_symbol'
            if len(items)!=len(set(items)):
                return 'repeated_refer_symbol'
        elif key==':include-macros':
            # Explicit compile-time sugar; never evaluate expressions in ns.
            if _sym(value) not in {'true','false'}:
                return 'non_literal_include_macros_boolean'
        elif key==':refer-macros':
            if value.kind!='vector' or not value.children:
                return 'unsupported_refer_macros_shape'
            names=[_sym(x) for x in value.children]
            if any(not simple_symbol(x) for x in value.children):
                return 'invalid_refer_macro_symbol'
            if len(set(names))!=len(names):
                return 'repeated_refer_macro_symbol'
        elif key==':rename':
            if value.kind!='map' or len(value.children)%2:
                return 'invalid_rename_map'
            items=[(_sym(value.children[j]),_sym(value.children[j+1]))
                   for j in range(0,len(value.children),2)]
            if any(not simple_symbol(x) for x in value.children):
                return 'invalid_rename_symbols'
            if len(items)!=len(set(a for a,b in items)):
                return 'ambiguous_rename_symbols'
    if ':as-alias' in seen and len(seen)>1:
        return 'ambiguous_alias_only_libspec'
    # Avoid guessing precedence between macro sugar specifiers when both are
    # given; the compiler may reject or normalize this differently by version.
    if ':include-macros' in seen and ':refer-macros' in seen:
        return 'combined_macro_sugar_not_verified'
    return None


def _unresolved_libspecs(forms: list[Form], platform: str = 'clj') -> list[dict]:
    from .clojure_libspecs import expanded_require_forms
    forms=expanded_require_forms(forms,platform,_libspec_issue)
    found=[]
    for f in forms:
        if not _top(f,'ns'):continue
        for clause in f.children[2:]:
            if clause.kind!='list' or not clause.children or _sym(clause.children[0])!=':require':
                continue
            for spec in clause.children[1:]:
                issue=_libspec_issue(spec,platform)
                if issue:
                    found.append({'reason':issue,'line':spec.line,'column':spec.col,
                                  'symbol':_sym(spec.children[0]) if spec.children else _sym(spec)})
    return found


def _namespace(forms: list[Form], platform: str = 'clj'):
    from .clojure_libspecs import expanded_require_forms
    forms=expanded_require_forms(forms,platform,_libspec_issue)
    statements=[f for f in forms if _top(f,'ns')]
    if len(statements)!=1:
        return None,{},[], ('multiple_ns_forms' if len(statements)>1 else 'no_ns_form')
    ns=statements[0]
    name=_sym(ns.children[1])
    from .clojure_libspecs import simple_symbol
    if not simple_symbol(ns.children[1]):
        return None,{},[],'invalid_ns_name'
    imports=[]; aliases={}; ambiguous_aliases=set()
    for entry in ns.children[2:]:
        if entry.kind!='list' or not entry.children or _sym(entry.children[0])!=':require':
            continue
        for spec in entry.children[1:]:
            if _libspec_issue(spec,platform):
                continue
            if spec.kind == 'atom':
                target=_sym(spec)
                if target and not target.startswith(':') and '/' not in target:
                    imports.append((target,spec.line,spec.col))
                continue
            if spec.kind!='vector' or not spec.children:
                # Prefix-lists, conditional requires and runtime requires
                # cannot be assumed to have one statically loaded target.
                continue
            target=_sym(spec.children[0])
            if not target or target.startswith(':') or '/' in target:
                continue
            # Inspect every option before creating a load or binding.
            # :as before :as-alias MUST NOT become a real require edge.
            opts = [_sym(part) for part in spec.children[1:]]
            if ':as-alias' in opts:
                continue  # never a loading require
            alias_options = [i for i, option in enumerate(opts) if option == ':as']
            if len(alias_options) > 1:
                continue  # ambiguous specification
            alias = None
            if alias_options:
                pos = alias_options[0]
                if pos+1 >= len(opts):
                    continue
                alias = opts[pos+1]
                if not alias or alias.startswith(':') or '/' in alias:
                    continue
            imports.append((target,spec.line,spec.col))
            if alias and alias in ambiguous_aliases:
                continue
            if alias and alias not in aliases:
                aliases[alias]=target
            elif alias:
                aliases.pop(alias,None)  # ambiguous alias, never choose first
                ambiguous_aliases.add(alias)
    return (name,ns.line,ns.col),aliases,imports,None


def _namespace_alias_guards(forms: list[Form], platform: str = 'clj') -> dict[str,str]:
    """Record ambiguous aliases and alias-only bindings *without* loading libs.

    A qualified head `x/f` must never turn into an external function edge
    when x appears in conflicting :as / :as-alias clauses. Unsupported
    libspecs contribute a blocker rather than being silently ignored.
    """
    from .clojure_libspecs import expanded_require_forms, invalid_prefix_aliases
    prefix_blockers=invalid_prefix_aliases(forms,platform,_libspec_issue)
    forms=expanded_require_forms(forms,platform,_libspec_issue)
    ns_forms=[f for f in forms if _top(f,'ns')]
    if len(ns_forms)!=1:
        return {}
    occurrences: dict[str,list[tuple[str,str,bool]]]={}
    for clause in ns_forms[0].children[2:]:
        if (clause.kind!='list' or not clause.children or
                _sym(clause.children[0]) not in {':require',':require-macros'}):
            continue
        section=_sym(clause.children[0])
        for spec in clause.children[1:]:
            if spec.kind!='vector' or not spec.children:
                continue
            namespace=_sym(spec.children[0])
            validated=_libspec_issue(spec, platform) is None
            options=spec.children[1:]
            for i, part in enumerate(options[:-1]):
                key=_sym(part)
                if key not in {':as',':as-alias'} or i%2!=0:
                    continue
                alias=_sym(options[i+1])
                if alias and not alias.startswith(':') and '/' not in alias:
                    occurrences.setdefault(alias,[]).append((section+key,namespace,validated))
    guarded={}
    for alias,entries in sorted(occurrences.items()):
        if len(entries)>1:
            guarded[alias]='ambiguous_namespace_alias_binding'
        elif not entries[0][2]:
            guarded[alias]='invalid_or_unverified_namespace_alias_spec'
        elif entries[0][0]!=':require:as':
            guarded[alias]='alias_only_or_compile_phase_binding_not_loaded_for_runtime_call'
    for alias in prefix_blockers:
        guarded[alias]='invalid_or_unverified_namespace_alias_spec'
    return guarded


def _callable_visibility(definition: Form) -> tuple[str,str]:
    """Literal Var privacy, including defn attribute-map merge precedence.

    Name metadata is merged first, then the leading attribute map and the
    optional trailing multi-arity map (core/defn). Expressions remain unknown;
    no project metadata is evaluated. A single-arity returned map is a body.
    """
    head=_sym(definition.children[0])
    named=definition.children[1]
    states=[]

    def private_state(meta):
        if meta.kind=='atom' and _sym(meta)==':private':
            return 'private'
        if meta.kind=='map':
            for i in range(0,len(meta.children)-1,2):
                if _sym(meta.children[i])==':private':
                    raw=_sym(meta.children[i+1])
                    return ('private' if raw=='true' else
                            'public' if raw in {'false','nil'} else 'unknown')
        return None

    for meta in named.metadata:
        state=private_state(meta)
        if state is not None:states.append(state)
    if head=='defn-':states.append('private')
    state=('unknown' if 'unknown' in states or len(set(states))>1 else
           'private' if 'private' in states else 'public')
    reason='private_var_metadata_or_defn_minus' if state=='private' else (
        'evaluated_or_conflicting_private_metadata' if state=='unknown' else
        'no_private_marker_in_supported_metadata_subset')
    if head in {'defn','defn-','defmacro'}:
        body=list(definition.children[2:])
        if body and body[0].kind=='string':body.pop(0)
        attributes=[]
        if body and body[0].kind=='map':attributes.append(body.pop(0))
        if body and body[0].kind=='list' and body[-1].kind=='map':
            attributes.append(body[-1])
        for meta in attributes:
            replacement=private_state(meta)
            if replacement is not None:
                state=replacement
                reason='literal_defn_attribute_map' if state!='unknown' else 'evaluated_defn_attribute_privacy'
    return state,reason


def _explicit_refers(forms: list[Form], platform: str = 'clj') -> tuple[dict[str, tuple[str,str] | None],list[str]]:
    """Extract only literal :refer entries from one ns form.

    Returned mappings are potential unqualified names, NOT confirmed vars.
    A None entry means an ambiguous local name; fail closed rather than use
    whichever library happened to appear first. :refer :all is unsupported.
    """
    from .clojure_libspecs import expanded_require_forms
    forms=expanded_require_forms(forms,platform,_libspec_issue)
    ns_forms=[f for f in forms if _top(f,'ns')]
    if len(ns_forms)!=1:
        return {},[]
    mapping:dict[str,tuple[str,str] | None]={}
    limitations=[]
    for clause in ns_forms[0].children[2:]:
        if clause.kind!='list' or not clause.children or _sym(clause.children[0])!=':require':
            continue
        for lib in clause.children[1:]:
            if _libspec_issue(lib,platform):
                limitations.append('unsupported_or_invalid_require_libspec')
                continue
            if lib.kind!='vector' or len(lib.children)<3:
                continue
            target=_sym(lib.children[0])
            if not target or target.startswith(':') or '/' in target:
                continue
            options=list(lib.children[1:])
            if ':as-alias' in [_sym(part) for part in options]:
                continue
            refers=[]
            rename={}
            i=0
            valid=True
            while i<len(options):
                key=_sym(options[i]);i+=1
                if not key.startswith(':') or i>=len(options):
                    valid=False;break
                val=options[i];i+=1
                if key==':refer':
                    if val.kind=='atom' and _sym(val)==':all':
                        limitations.append('require_refer_all_requires_runtime_export_enumeration')
                        valid=False;break
                    if val.kind not in ('vector','list') or refers:
                        valid=False;break
                    refers=[_sym(child) for child in val.children]
                    if not refers or any(not item or item.startswith(':') or '/' in item for item in refers):
                        valid=False;break
                elif key==':rename':
                    if val.kind!='map' or len(val.children)%2:
                        valid=False;break
                    for j in range(0,len(val.children),2):
                        src=_sym(val.children[j]);dst=_sym(val.children[j+1])
                        if not src or not dst or src.startswith(':') or dst.startswith(':') or '/' in src or '/' in dst:
                            valid=False;break
                        rename[src]=dst
                elif key in (':as',':reload',':reload-all',':verbose',
                             ':include-macros', ':refer-macros'):
                    # Explicit macro sugar does not change a *runtime* :refer.
                    # Macro references are handled in the separate phase.
                    continue
                else:
                    # An unrecognized require option can change its meaning.
                    valid=False;break
            if not valid:
                limitations.append('require_options_not_fully_resolved')
                continue
            for name in refers:
                local=rename.get(name,name)
                candidate=(target,name)
                if local in mapping:
                    mapping[local]=None
                    limitations.append('duplicate_or_ambiguous_refer_local_name')
                else:
                    mapping[local]=candidate
    return mapping,sorted(set(limitations))


def _definition(f: Form):
    if f.kind!='list' or len(f.children)<2:return None
    head=_sym(f.children[0])
    types={'def':'variable','defonce':'variable','defn':'function','defn-':'function',
           'defmacro':'macro','defmulti':'multimethod','defprotocol':'protocol',
           'defrecord':'record','deftype':'type','definterface':'interface'}
    kind=types.get(head)
    if kind is None:return None
    name_form=f.children[1]
    name=_sym(name_form)
    from .clojure_libspecs import simple_symbol
    if not simple_symbol(name_form):return None
    # A bare `(defn f)` is incomplete; never publish a callable symbol.
    if head in {'def','defonce'}:
        from .clojure_var_initializer import parse_variable_initializer
        if parse_variable_initializer(f).status == 'unsupported':
            return None  # malformed literal def cannot prove a declaration
    if head in {'defn','defn-','defmacro'}:
        body=list(f.children[2:])
        if body and body[0].kind=='string':
            body.pop(0)
        if body and body[0].kind=='map':
            body.pop(0)
        if not body or not (body[0].kind=='vector' or
            (body[0].kind=='list' and body[0].children and
             body[0].children[0].kind=='vector')):
            return None
    return name,kind,name_form


MAX_DECLARATION_MEMBERS = 128


def _literal_declaration_members(definition: Form, kind: str):
    """Index only bounded, unambiguous field/method *syntax*.

    A protocol method is a declared polymorphic Var, not proof of any
    implementation or dispatch target. Record/type fields are declarations,
    not proof that a field access resolves to this field. Complex/unverified
    signatures remain outside the declared subset.
    """
    from .clojure_libspecs import simple_symbol
    items=[]
    if kind == 'protocol':
        seen=set()
        for body in definition.children[2:]:
            if body.kind != 'list':
                continue  # e.g. docstring, :extend-via-metadata option
            if len(body.children)<2 or not simple_symbol(body.children[0]):
                return [],'protocol_method_signature_unverified'
            name_form=body.children[0]
            name=name_form.value
            if name in seen:
                return [],'duplicate_literal_protocol_method'
            seen.add(name)
            signatures=body.children[1:]
            if signatures and signatures[-1].kind=='string':
                signatures=signatures[:-1]
            if (not signatures or len(signatures)>MAX_DECLARATION_MEMBERS or
                    any(v.kind!='vector' or not v.children or
                        any(not simple_symbol(part) for part in v.children)
                        for v in signatures)):
                return [],'protocol_method_signature_unverified'
            items.append((name,'protocol_method_declaration',name_form))
            if len(items)>MAX_DECLARATION_MEMBERS:
                return [],'protocol_member_limit'
    elif kind == 'interface':
        # definterface declares JVM interface methods, not runtime Var calls.
        # A Java instance method's argument vector excludes the receiver;
        # a deftype implementation has `this` as its first parameter.
        signatures=set()
        for body in definition.children[2:]:
            if (body.kind!='list' or len(body.children)!=2 or
                    not simple_symbol(body.children[0]) or
                    body.children[1].kind!='vector' or
                    any(not simple_symbol(x) for x in body.children[1].children)):
                return [],'interface_method_signature_unverified'
            marker=(body.children[0].value,len(body.children[1].children))
            if marker in signatures:
                return [],'duplicate_interface_method_arity'
            signatures.add(marker)
            items.append((body.children[0].value,'interface_method_declaration',
                          body.children[0]))
            if len(items)>MAX_DECLARATION_MEMBERS:
                return [],'interface_member_limit'
    elif kind in {'type','record'}:
        if len(definition.children)<3 or definition.children[2].kind!='vector':
            return [],'record_type_field_vector_not_verified'
        seen=set()
        for field in definition.children[2].children:
            if not simple_symbol(field):
                return [],'record_type_field_not_literal_symbol'
            if field.value in seen:
                return [],'duplicate_literal_record_type_field'
            seen.add(field.value)
            items.append((field.value,'record_field_declaration' if kind=='record'
                          else 'type_field_declaration',field))
            if len(items)>MAX_DECLARATION_MEMBERS:
                return [],'record_type_field_limit'
    return items,None


def _source_platform(path: str) -> str:
    """A .cljc file has shared syntax; target platform is not yet selected."""
    return {'.clj':'clj','.cljs':'cljs','.cljc':'shared'}.get(Path(path).suffix.lower(),'unknown')


def _select_namespace_targets(nodes: list[dict], source_path: str) -> list[dict]:
    """Select only unambiguous candidate targets in the matching platform.

    The presence of .class files and runtime classpath order is not checked,
    so this remains candidate evidence, NOT a resolved namespace load.
    """
    platform=_source_platform(source_path)
    by_platform: dict[str,list[dict]]={}
    for node in nodes:
        by_platform.setdefault(_source_platform(node['path']),[]).append(node)
    if platform in {'clj','cljs'}:
        preferred=by_platform.get(platform,[])
        selected=preferred if preferred else by_platform.get('shared',[])
        return selected if len(selected)==1 else []
    if platform=='shared':
        # No .clj/.cljs branch selection or explicit build context exists.
        # Publishing a single target for both would fabricate precision.
        if any(by_platform.get(p) for p in ('clj','cljs','unknown')):
            return []
        shared=by_platform.get('shared',[])
        return shared if len(shared)==1 else []
    return []


# Only these are Clojure compiler special forms (not arbitrary macros).
# These nodes describe observed lexical syntax, NOT a proven CFG or runtime path.
STRUCTURAL_FORMS = frozenset({'if','do','let','loop','recur','fn','try',
                              'catch','finally','throw','var','set!','new',
                              'monitor-enter','monitor-exit'})


def _walk_special_forms(form: Form):
    """Read-only lexical structure; no expansion/evaluation or computed control flow."""
    if form.kind == 'opaque':
        return
    if form.kind == 'list' and form.children:
        head = _sym(form.children[0])
        if head in {'quote','clojure.core/quote','comment','clojure.core/comment'}:
            return
        if head in OPAQUE_BINDING_MACROS:
            # These macros may treat arguments as data or introduce lexical
            # bindings; do not publish nested syntax as executable structure.
            return
        if head in {'case','clojure.core/case'}:
            # Test constants in case are DATA, not executable expressions.
            if len(form.children)>1:
                yield from _walk_special_forms(form.children[1])
            for i in range(3,len(form.children),2):
                yield from _walk_special_forms(form.children[i])
            if len(form.children)>2 and len(form.children)%2==1:
                yield from _walk_special_forms(form.children[-1])
            return
        if head in STRUCTURAL_FORMS:
            yield form, head
    for child in form.children:
        yield from _walk_special_forms(child)


def _calls(form: Form):
    if form.kind == 'opaque':return
    if form.kind=='list' and form.children:
        # 'quote' is a SPECIAL FORM; its body is data, not an executable call.
        # Clojure's built-in comment macro likewise ignores its arguments.
        # Never turn their nested lists into call evidence.
        if _sym(form.children[0]) in {'quote', 'clojure.core/quote',
                                      'comment', 'clojure.core/comment'}:
            return
        head=form.children[0]
        sym=_sym(head)
        if sym and '/' in sym and not sym.startswith(('/', ':')):
            yield head
        if sym in {'case', 'clojure.core/case'}:
            # The official case macro treats each test as a compile-time
            # literal, NOT as an evaluated expression. A test written as a
            # list is a collection of constants, not a runtime function call.
            # Only the dispatch expression, selected result expressions, and
            # optional default can contain actual call sites.
            items=form.children
            if len(items)>1:
                yield from _calls(items[1])  # dispatch expression
            for i in range(3,len(items),2):
                yield from _calls(items[i])  # matching result expression
            if len(items)>2 and len(items)%2==1:
                yield from _calls(items[-1])  # optional default expression
            return
        for item in form.children[1:]:
            yield from _calls(item)
        return
    for child in form.children:
        yield from _calls(child)


def analyze_clojure(root: Path, candidates: dict, _graph: dict) -> tuple[dict,list[dict]]:
    from .clojure_protocol_implementations import (
        parse_direct_implementations, parse_literal_extend, walk_reify_sites)
    root=root.resolve()
    files={};warnings=[]
    paths=sorted(path for path,v in candidates.items() if v.get('language')=='clojure')
    for path in paths:
        p=root/path
        try:
            if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
                raise ReaderError('missing_symlink_or_outside_root')
            if p.stat().st_size>MAX_BYTES:
                raise ReaderError('source_exceeds_limit')
            raw=p.read_bytes()
            text=raw.decode('utf-8-sig')
            platform=_source_platform(path)
            tokens=lex(text,platform=platform)
            forms=_Parser(tokens,platform=platform).all()
            from .clojure_reader_grammar import classify_source
            grammar_evidence = classify_source(text, platform=(
                'cljc' if platform == 'shared' else platform),
                context='project_source')
            if grammar_evidence['status'] == 'rejected':
                raise ReaderError('reader_grammar_rejected:' + str(
                    grammar_evidence.get('reasons') or grammar_evidence.get('reason'))[:120])
            limitations=_reader_limitations(tokens, forms)
            # .cljc has multiple platform-specific reader views. Keep them
            # separate as descriptive candidate evidence; NEVER promote them
            # to proved cross-file edges or mix :clj and :cljs dependencies.
            projections = {}
            projected_cache = {}
            if platform == 'shared':
                from .clojure_conditionals import ProjectionUnavailable, project
                for target in ('clj','cljs'):
                    try:
                        projection=project(forms,target)
                        projected=list(projection.forms)
                        projected_cache[target]=tuple(projected)
                        candidate_ns=_namespace(projected,target)
                        candidate_defs=[]
                        from .clojure_multimethod import parse_method_site
                        projected_refers=_explicit_refers(projected,target)[0]
                        from .clojure_protocol_extensions import parse_extension_methods
                        for projected_form in projected:
                            recognized=_definition(projected_form)
                            definitions=[recognized] if recognized else []
                            if recognized is None:
                                method_site=parse_method_site(projected_form,
                                    projected,projected_refers,platform=target)
                                if method_site is not None:
                                    definitions.append((method_site.name_form.value,
                                        'multimethod_method',method_site.name_form))
                                extension_sites=parse_extension_methods(projected_form,
                                    projected,projected_refers,platform=target)
                                if extension_sites:
                                    definitions.extend((site.method.children[0].value,
                                        'protocol_implementation_method',site.method.children[0])
                                        for site in extension_sites)
                                literal_sites=parse_literal_extend(projected_form,
                                    projected,projected_refers,platform=target)
                                if literal_sites:
                                    definitions.extend((site.name_form.value,
                                        'protocol_implementation_method',site.name_form)
                                        for site in literal_sites)
                            direct_sites=parse_direct_implementations(projected_form,
                                projected,projected_refers,platform=target)
                            if direct_sites:
                                definitions.extend((site.name_form.value,
                                    'protocol_implementation_method',site.name_form)
                                    for site in direct_sites)
                            for reify_site in walk_reify_sites(projected_form,
                                    projected,projected_refers,platform=target):
                                definitions.append((reify_site.name_form.value,
                                    'protocol_implementation_method',reify_site.name_form))
                            for name,kind,decl in definitions:
                                candidate_defs.append({'name':name,'kind':kind,
                                                       'line':decl.line,'column':decl.col})
                        projections[target]={
                            'status':'candidate_not_semantically_verified',
                            'read_mode':':read-cond :allow',
                            'form_count':len(projected),
                            'selected_conditionals':projection.selected_conditionals,
                            'omitted_conditionals':projection.omitted_conditionals,
                            'spliced_elements':projection.spliced_elements,
                            'namespace':candidate_ns[0][0] if candidate_ns[0] else None,
                            'namespace_line':candidate_ns[0][1] if candidate_ns[0] else None,
                            'namespace_column':candidate_ns[0][2] if candidate_ns[0] else None,
                            'namespace_issue':candidate_ns[3],
                            'candidate_requires':[
                                {'namespace':name,'line':ln,'column':col}
                                for name,ln,col in candidate_ns[2]],
                            'candidate_definitions':candidate_defs[:200],
                            'definitions_truncated':len(candidate_defs)>200,
                        }
                    except ProjectionUnavailable as err:
                        projections[target]={'status':'unavailable',
                                             'reason':str(err)[:180]}
            from .clojure_macro_phase import macro_require_specs
            macro_source = (projections.get('cljs',{}) if platform=='shared' else {})
            # For .cljc, derive only from the actually projected cljs forms;
            # never borrow JVM-only macro clauses.
            if platform == 'shared' and macro_source.get('status')=='candidate_not_semantically_verified':
                macro_forms=projected_cache['cljs']
                macro_specs=macro_require_specs(macro_forms,platform='cljs',validation=_libspec_issue)
            elif platform == 'cljs':
                macro_specs=macro_require_specs(forms,platform='cljs',validation=_libspec_issue)
            else:
                macro_specs={'status':'unavailable' if platform=='shared' else 'not_applicable',
                             'requires':[],'limitations':['cljs_projection_unavailable'] if platform=='shared' else []}
            files[path]={'sha':hashlib.sha256(raw).hexdigest(),
                         'macro_require_specs':macro_specs,
                         'lines':text.splitlines(), 'text':text, 'forms':forms,
                         'reader_limitations':limitations,
                         'reader_grammar_evidence':grammar_evidence,
                         'platform_projections':projections,
                         '_platform_projected_forms':projected_cache,
                         'namespace':_namespace(forms,platform),
                         'alias_guards':_namespace_alias_guards(forms,platform),
                         'referred_names':_explicit_refers(forms,platform),
                         'unresolved_libspecs':_unresolved_libspecs(forms,platform)}
        except (ReaderError,UnicodeError,OSError) as exc:
            warnings.append({'path':path,'reason':'clojure_reader_unavailable_or_invalid_subset',
                             'details':str(exc)[:170]})
    # Explicit, optional compiler-distribution reader validation. Only these
    # locally selected JARs enter the reference subprocess classpath; source
    # text is stdin data, never loaded as a namespace or executed.
    oracle = _graph.get('_atlas_clojure_reader_jar_dir') if isinstance(_graph, dict) else None
    if oracle and files:
        from .clojure_official_reader import validate_sources
        source_requests=[]
        for candidate, record in sorted(files.items()):
            platform=_source_platform(candidate)
            if platform=='shared':
                source_requests.extend(((candidate+'#clj','cljc',record['text']),
                                        (candidate+'#cljs','cljs',record['text'])))
            else:
                source_requests.append((candidate,platform,record['text']))
        observations={}
        for begin in range(0,len(source_requests),100):
            observations.update(validate_sources(source_requests[begin:begin+100],Path(oracle)))
        # An explicitly chosen official reader is an authority on whether a
        # source is syntactically readable. If it rejects a source, the
        # structural subset must NOT continue to emit plausible-looking Var
        # or require edges. The ordinary inventory still includes the file.
        # For .cljc both platform views must be independently readable;
        # retaining only one shared structural tree would conflate platforms.
        for candidate,record in list(files.items()):
            platform=_source_platform(candidate)
            if platform=='shared':
                result={'clj':observations[candidate+'#clj'],
                        'cljs':observations[candidate+'#cljs']}
            else:
                result={platform:observations[candidate]}
            record['official_reader_evidence']=result
            rejected=[key for key,observation in result.items()
                      if observation['status']!='accepted']
            # The structural subset may also inadvertently drop a top-level
            # form in an officially accepted native source. Never promote
            # those incomplete trees into semantic evidence.
            count_mismatch=(platform!='shared' and not rejected and
                            result[platform]['forms_read']!=len(record['forms']))
            if rejected or count_mismatch:
                warnings.append({'path':candidate,
                    'reason':('official_reader_rejected_source' if rejected else
                              'structural_reader_top_level_count_mismatch'),
                    'details':','.join(rejected) if rejected else 'official_reader_count_differs_from_structural_subset'})
                del files[candidate]
    # Bounded, source-only JVM :refer :all. Never load a project ns to
    # enumerate ns-publics; exact classpath and platform guards remain active.
    from .clojure_project_refer_all import materialize as _project_refer_all
    _project_refer_all(files, namespace=_namespace, definition=_definition,
        visibility=_callable_visibility, libspec_issue=_libspec_issue,
        source_platform=_source_platform)
    # Optional trusted standard-library oracle: proves clojure.core Var
    # identity against the selected pinned local JARs. This is NOT project
    # compiler analysis, arbitrary user macro execution or CLJS resolution.
    core_oracle_evidence={}
    if oracle and files:
        from .clojure_core_evidence import discover_core_sites
        from .clojure_core_oracle import core_oracle
        plans={}
        requests=[]
        for path,record in sorted(files.items()):
            if _source_platform(path)!='clj':continue
            plan=discover_core_sites(record['forms'],record['text'],record['referred_names'][0])
            plans[path]=plan
            for item,name,text_source in plan['macros']:
                requests.append((path,item,name,text_source))
        names=sorted({site['name'] for plan in plans.values() for site in plan['sites']})
        # Multiple fixture files may share a macro invocation. Preserve
        # source-specific identity and don't guess when resource limits apply.
        if len(names)+len(requests)<=2000:
            observation=core_oracle(names,[(name,src) for _,_,name,src in requests],Path(oracle))
            for path,plan in plans.items():
                confirmed=[]
                for site in plan['sites']:
                    record=observation['vars'].get(site['name'])
                    if record and record['status']=='found':
                        confirmed.append({**site,'resolution':'clojure_core_var_identity_runtime_backed',
                                          'var':record['qualified_name'],
                                          'macro':record['macro'],
                                          'compiler_var_tag':record['compiler_var_tag'],
                                          'declared_fixed_arities':record['declared_fixed_arities'],
                                          'declared_variadic_min_arity':record['declared_variadic_min_arity'],
                                          'arity_metadata_available':record['arity_metadata_available'],
                                          'arity_evidence':'trusted_core_var_meta_arglists_not_project_call_check'})
                core_oracle_evidence[path]={'status':'trusted_clojure_core_only',
                    'provenance':observation['provenance'],'var_sites':confirmed,
                    'macro_expansions':[], 'limitations':plan['limitations']}
            for i,(path,site,name,src) in enumerate(requests):
                expansion=observation['expansions'][i]
                syntax_shape=[]
                if expansion['status']=='expanded' and expansion.get('expanded_syntax'):
                    try:
                        # These locations refer to GENERATED syntax, never to
                        # original source coordinates. Only the original macro
                        # head has a verified user source position.
                        parsed=read_forms(expansion['expanded_syntax'])
                        if len(parsed)==1:
                            syntax_shape=[head for special,head in _walk_special_forms(parsed[0])][:80]
                    except (ReaderError,ValueError):
                        pass
                core_oracle_evidence[path]['macro_expansions'].append({
                    'macro':name,'containing_declaration':site['containing_declaration'],
                    'location':site['location'],**expansion,
                    'generated_core_special_form_shapes':syntax_shape,
                    'generated_source_locations':False,
                    'resolution':'trusted_core_macro_expansion_not_project_compiler_binding'})
        else:
            for path,plan in plans.items():
                core_oracle_evidence[path]={'status':'unavailable',
                                           'reason':'core_oracle_global_request_limit',
                                           'limitations':plan['limitations']}
    # CLJS gets its OWN official compiler index: the trusted cljs/core.cljs
    # from the supplied JAR is analyzed once, never a user namespace.  This
    # must not be confused with the Clojure/JVM Var registry above.
    if oracle and files:
        from .clojure_core_evidence import discover_core_sites
        from .clojurescript_core_oracle import cljs_core_oracle
        cljs_plans={}
        for path,record in sorted(files.items()):
            if _source_platform(path)!='cljs':continue
            cljs_plans[path]=discover_core_sites(record['forms'],record['text'],record['referred_names'][0])
        cljs_names=sorted({site['name'] for plan in cljs_plans.values() for site in plan['sites']})
        if cljs_plans and len(cljs_names)<=2000:
            cljs_observed=cljs_core_oracle(cljs_names,Path(oracle))
            for path,plan in cljs_plans.items():
                matched=[]
                for site in plan['sites']:
                    item=cljs_observed['vars'].get(site['name'])
                    if item and item['status']=='found':
                        matched.append({**site,'resolution':'cljs_core_compiler_namespace_identity',
                             'var':item['qualified_name'],'macro':item['macro'],
                             'fn_var':item['fn_var'],'compiler_tag':item['compiler_tag'],
                             'declared_fixed_arities':item['declared_fixed_arities'],
                             'declared_variadic_min_arity':item['declared_variadic_min_arity'],
                             'arity_metadata_available':item['arity_metadata_available'],
                             'arity_evidence':'trusted_cljs_core_compiler_arglists_not_project_call_check'})
                core_oracle_evidence[path]={'status':'trusted_cljs_core_compiler_index',
                    'provenance':cljs_observed['provenance'],'var_sites':matched,
                    'macro_expansions':[],
                    'limitations':plan['limitations']+[
                         {'reason':'project_cljs_macro_expansion_not_run'}]}
        elif cljs_plans:
            for path,plan in cljs_plans.items():
                core_oracle_evidence[path]={'status':'unavailable',
                   'reason':'cljs_core_oracle_name_limit','limitations':plan['limitations']}
    # Stage 41: pure, bounded source-index lookups for each platform. These
    # results are scoped candidate EVIDENCE, never promoted to verified edges.
    from .clojure_namespace_evidence import build_platform_namespace_evidence
    from .clojure_source_context import build_source_contexts, select_namespace_paths, source_path_evidence
    namespace_names=set()
    for record in files.values():
        ns_record=record['namespace'][0]
        if ns_record:namespace_names.add(ns_record[0])
        # A required namespace can be supplied exclusively by an external JAR.
        # Include literal imports in exact-resource classpath probes.
        for imported, _ln, _col in record['namespace'][2]:
            namespace_names.add(imported)
        for projection in record.get('platform_projections',{}).values():
            if projection.get('namespace'):namespace_names.add(projection['namespace'])
    source_contexts=build_source_contexts(root,files,namespace_names,
        selected_aliases=_graph.get('_atlas_clojure_cli_aliases',()) if isinstance(_graph,dict) else ())
    # A caller may explicitly provide a precomputed classpath snapshot.
    # Do not invoke tools.deps or package managers from Atlas; recorded paths
    # and JAR checksums provide reproducible local resource evidence only.
    snapshot_file = _graph.get('_atlas_clojure_classpath_snapshot') if isinstance(_graph,dict) else None
    if snapshot_file and files:
        from .clojure_classpath_snapshot import inspect_snapshot
        evidence = inspect_snapshot(Path(snapshot_file), namespace_names, root,
            selected_aliases=_graph.get('_atlas_clojure_cli_aliases',()))
        for source, context in list(source_contexts.items()):
            source_contexts[source] = {**context, 'externally_supplied_classpath':evidence}
    for path,record in files.items():record['project_source_context']=source_contexts[path]
    namespace_evidence=build_platform_namespace_evidence(files,source_contexts)
    from .clojure_macro_phase import build_macro_phase_evidence
    macro_phase_evidence=build_macro_phase_evidence(files,namespace_evidence)
    # Stage 53 source-anchored token CST: independently bounded and always
    # separate from official reader acceptance and compiler semantic claims.
    from .clojure_cst import build_source_cst
    for candidate, record in files.items():
        record['source_cst'] = build_source_cst(
            record['text'], lex(record['text'], platform=_source_platform(candidate)),
            source_sha256=record['sha'], platform=_source_platform(candidate),
            official_reader_evidence=record.get('official_reader_evidence'))
        from .clojure_source_ast import source_form_ast
        record['source_form_ast']=source_form_ast(
            record['text'], record['forms'],record['sha'],_source_platform(candidate))
        if _source_platform(candidate)=='shared':
            record['cljc_projected_form_asts']={
                language:source_form_ast(record['text'], projected,record['sha'],language)
                for language,projected in sorted(record['_platform_projected_forms'].items())}

    graphs={};ns_index={};def_index={}
    pending_requires=[];pending_calls=[];pending_referred_calls=[];pending_alias_issues=[];pending_methods=[];pending_extensions=[]
    for path,data in sorted(files.items()):
        sha,lines,forms=data['sha'],data['lines'],data['forms']
        ns,aliases,imports,problem=data['namespace']
        referred,refer_warnings=data['referred_names']
        from .clojure_core_context import shadowed_core_names, core_macro_aliases
        core_shadowed=shadowed_core_names(data['forms'],referred)
        macro_aliases=core_macro_aliases(data['forms'])
        # Resolve the bounded, source-only core macro spellings ONCE per file,
        # not once for every defn in a large Clojure project.
        from .clojure_short_circuit import candidates as short_circuit_candidates
        short_circuit = short_circuit_candidates(
            forms, platform=_source_platform(path), referred=referred,
            definition=_definition, top=_top, sym=_sym)
        fnode={'id':_file_id(path),'path':path,'name':Path(path).name,'kind':'file','line':1,'confidence':'verified_file'}
        from .clojure_classpath_snapshot import public_snapshot_evidence
        public_context={k:v for k,v in source_contexts[path].items() if not k.startswith('_')}
        if 'externally_supplied_classpath' in public_context:
            public_context['externally_supplied_classpath']=public_snapshot_evidence(
                public_context['externally_supplied_classpath'],root)
        g={'nodes':[fnode],'edges':[],'hash':sha,'parser':'clojure_structural_forms_subset',
           'reader_limitations':data['reader_limitations'],
           'reader_grammar_evidence':data['reader_grammar_evidence'],
           'clojure_source_cst':data['source_cst'],
           'clojure_source_form_ast':data['source_form_ast'],
           'cljc_projected_form_asts':data.get('cljc_projected_form_asts',{}),
           'official_reader_evidence':data.get('official_reader_evidence', {'status':'not_requested'}),
           'core_var_oracle_evidence':core_oracle_evidence.get(path, {'status':'not_requested'}),
           'declaration_limitations':[],
           'clojure_semantic_analysis':{},
           'platform_projections':data['platform_projections'],
           'clojure_project_context':public_context,
           'platform_namespace_evidence':namespace_evidence.get(path,{}),
           'platform_macro_phase_evidence':macro_phase_evidence.get(path,{}),
           'semantic_completeness':'unverified_partial',
           'source_coordinates':'one_based_unicode_codepoints_end_exclusive',
           'unqualified_symbol_binding':'explicit_refer_subset_lexical_guard_not_certified',
           'lexical_scope_limitations':refer_warnings,
           'project_refer_all_evidence':data.get('refer_all_evidence',[]),
           'lexical_scope_suppression_count':0,
           'lexical_scope_suppression_examples':[],
           'qualified_scope_suppression_count':0,
           'qualified_scope_suppression_examples':[],
           'unresolved_references':[],
           'unresolved_reference_count':0,
           'platform_scope':_source_platform(path),
           'namespace_alias_guard_reasons':data['alias_guards'],
           'private_var_guard':'literal_private_metadata_subset_not_compiler_certified',
           'caution':'Offline balanced Clojure form reader, selected top-level declarations and literal require aliases. '
             'Optional vetted Clojure core macro expansion and core Var metadata only; '
             'no project macro execution, complete type semantics or runtime call guarantees. '
             'Literal def/defonce initializers are candidate syntax; defonce may skip evaluation.'}
        graphs[path]=g
        if problem and problem != 'no_ns_form':
            warnings.append({'path':path,'reason':'clojure_namespace_unresolved','details':problem})
        def ev(line,col=0, *, end_line=None,end_column=None,
               _path=path, _sha=sha, _lines=lines):
            evidence={'path':_path,'line':line,'column':col,'source_sha256':_sha,
                    'snippet':_lines[line-1].strip()[:200] if 0<line<=len(_lines) else ''}
            if end_line is not None and end_column is not None:
                evidence.update(end_line=end_line,end_column=end_column)
            return evidence
        def node(parent,name,qual,kind,part, *, relation='declares'):
            n={'id':_node_id(path,qual,part.line,part.col), 'path':path,'name':name,
               'qualname':qual,'kind':kind,'line':part.line,'column':part.col,
               'parent':parent['id'],'confidence':'parsed_structural_subset'}
            if part.end_line is not None and part.end_col is not None:
                n.update(end_line=part.end_line,end_column=part.end_col)
            g['nodes'].append(n)
            g['edges'].append({'source':parent['id'],'target':n['id'],'source_path':path,
                'target_path':path,'relation':relation,'resolution':'parsed_structural_subset',
                'analyzer':'clojure-form-reader',
                'evidence':ev(part.line,part.col,end_line=part.end_line,
                              end_column=part.end_col)})
            return n
        ns_node=None
        if ns:
            name,ln,col=ns
            nsform=next(f for f in forms if _top(f,'ns'))
            ns_node=node(fnode,name,name,'namespace',nsform.children[1])
            ns_index.setdefault(name,[]).append(ns_node)
            for target,line,col in imports:
                pending_requires.append((path,ns_node,target,line,col))
        special_seen=0
        from .clojure_multimethod import parse_method_site
        from .clojure_protocol_extensions import parse_extension_methods
        from .clojure_protocol_implementations import (
            parse_direct_implementations, parse_literal_extend, walk_reify_sites)

        def publish_protocol_sites(sites, parent):
            """One source graph pipeline for all literal implementation shapes."""
            from .clojure_semantics import analyze_declaration
            for site in sites:
                member_form = site.name_form
                member = member_form.value
                proto_name = site.protocol.value
                type_name = (site.type_form.value if site.type_form is not None
                             else f'<anonymous@{site.owner_form.line}:{site.owner_form.col}>')
                source_qual = ((ns[0]+'/' if ns else '')+
                    f'implementation:{site.construction}:{proto_name}:{type_name}:'
                    f'{member}@{member_form.line}:{member_form.col}')
                implementation=node(parent, member, source_qual,
                    'protocol_implementation_method', member_form)
                implementation['source_platform']=_source_platform(path)
                implementation['protocol_source_name']=proto_name
                implementation['type_source_name']=type_name
                implementation['implementation_syntax']=site.construction
                implementation['implementation_kind']=site.implementation_kind
                if site.implementation_kind=='direct_source_method':
                    fixed=[]
                    variadic=None
                    for params,_body in site.arities:
                        symbols=[part.value for part in params.children]
                        if '&' in symbols:
                            variadic=symbols.index('&')
                        else:
                            fixed.append(len(symbols))
                    implementation['declared_fixed_arities']=sorted(set(fixed))
                    implementation['declared_variadic_min_arity']=variadic
                    implementation['arity_evidence']='literal_implementation_source_only'
                implementation['semantic_resolution']=(
                    'literal_protocol_implementation_source_candidate_not_dispatch')
                implementation['type_binding_confirmed']=False
                implementation['protocol_binding_confirmed']=False
                implementation['runtime_dispatch_confirmed']=False
                implementation['method_source']=ev(site.method.line,site.method.col,
                    end_line=site.method.end_line,end_column=site.method.end_col)
                if site.type_form is not None:
                    implementation['type_source']=ev(site.type_form.line,
                        site.type_form.col,end_line=site.type_form.end_line,
                        end_column=site.type_form.end_col)
                if site.implicit_fields:
                    implementation['captured_type_fields']=sorted(site.implicit_fields)
                from .clojure_type_hints import literal_type_tag
                site_hints=[]
                for params,_expressions in site.arities:
                    for position,param in enumerate(params.children):
                        if (tag:=literal_type_tag(param)) is not None:
                            site_hints.append({'parameter_position':position,
                                'parameter':param.value if param.kind=='atom' else None,
                                'type_tag':tag})
                if site_hints:
                    implementation['literal_parameter_type_hints']=site_hints
                # JVM Object/interface methods are not necessarily project
                # protocols. Their syntax is visible, without a spurious edge.
                if proto_name not in {'Object','java.lang.Object','js/Object'}:
                    pending_extensions.append((path,implementation,
                                               site.protocol,member_form))
                elif len(g['declaration_limitations'])<MAX_UNRESOLVED_EXAMPLES:
                    g['declaration_limitations'].append({
                        'reason':'host_object_method_not_project_protocol',
                        'declaration':source_qual,'evidence':ev(member_form.line,member_form.col)})
                if site.implementation_kind=='literal_method_map_value_candidate':
                    # `(extend ... {:f expression})` evaluates its mapping.
                    # The value need not be a function and is not a method body.
                    implementation['mapping_value_expression_kind']=site.method.kind
                    continue
                synthetic=Form('list','(',(
                    Form('atom','defn',(),site.method.line,site.method.col),
                    member_form,*site.method.children[1:]),
                    site.method.line,site.method.col,(),
                    site.method.end_line,site.method.end_col)
                g['clojure_semantic_analysis'][source_qual]=analyze_declaration(
                    synthetic,namespace=ns[0] if ns else '',
                    phase='runtime_candidate',core_short_circuit=short_circuit,
                    implicit_field_forms=site.field_forms)
                function_form=Form('list','(',(
                    Form('internal_function_scope','fn*',(),site.method.line,site.method.col),
                    *site.method.children[1:]),site.method.line,site.method.col)
                if referred:
                    sites_found,suppressions=referred_call_sites((function_form,),
                        frozenset(referred),core_shadowed=core_shadowed,
                        macro_aliases=macro_aliases,
                        implicit_bindings=site.implicit_fields)
                    for call_site in sites_found:
                        pending_referred_calls.append((path,implementation,
                            call_site.local_name,referred[call_site.local_name],
                            call_site.head,'runtime_candidate',call_site.argument_count))
                    g['lexical_scope_suppression_count']+=len(suppressions)
                sites_found,suppressions=referred_call_sites((function_form,),
                    frozenset(),include_qualified=True,core_shadowed=core_shadowed,
                    macro_aliases=macro_aliases,
                    implicit_bindings=site.implicit_fields)
                for call_site in sites_found:
                    prefix,name=call_site.head.value.split('/',1)
                    if not name:continue
                    if prefix in data['alias_guards']:
                        pending_alias_issues.append((path,call_site.head,
                            data['alias_guards'][prefix]))
                    elif prefix in aliases or '.' in prefix or prefix in namespace_names:
                        pending_calls.append((path,implementation,
                            aliases.get(prefix,prefix),name,call_site.head,
                            'runtime_candidate',call_site.argument_count))
                g['qualified_scope_suppression_count']+=len(suppressions)

        for f in forms:
            # `reify` is an expression (often inside a function), not a
            # namespace declaration. Keep its methods source-visible.
            nested_reify=tuple(walk_reify_sites(f,forms,referred,
                platform=_source_platform(path)))
            if nested_reify:
                publish_protocol_sites(nested_reify,ns_node or fnode)
            parsed=_definition(f)
            if not parsed:
                extension_methods=parse_extension_methods(f, forms, referred,
                    platform=_source_platform(path))
                literal_extend=parse_literal_extend(f,forms,referred,
                    platform=_source_platform(path))
                if extension_methods is not None or literal_extend is not None:
                    from .clojure_protocol_implementations import SourceImplementation
                    extended=tuple(SourceImplementation(
                        site.protocol,site.type_form,site.method,
                        site.method.children[0],site.arities,site.expansion_macro,
                        'direct_source_method',f)
                        for site in (extension_methods or ()))
                    publish_protocol_sites((*extended,*(literal_extend or ())),
                                           ns_node or fnode)
                    continue
                method = parse_method_site(f, forms, referred,
                    platform=_source_platform(path))
                if method is None:
                    continue
                method_name = method.name_form.value
                method_qual = ((ns[0] + '/' if ns else '') +
                    f'defmethod:{method_name}@{f.line}:{f.col}')
                implementation = node(ns_node or fnode, method_name,
                                      method_qual, 'multimethod_method',
                                      method.name_form)
                implementation['semantic_resolution'] = (
                    'multimethod_source_registration_not_runtime_dispatch')
                implementation['source_platform'] = _source_platform(path)
                implementation['dispatch_evidence'] = {
                    'expression_kind': method.dispatch_form.kind,
                    'source': ev(method.dispatch_form.line, method.dispatch_form.col,
                        end_line=method.dispatch_form.end_line,
                        end_column=method.dispatch_form.end_col),
                    'runtime_dispatch_confirmed': False,
                    'macroexpansion_performed': False}
                from .clojure_semantics import analyze_declaration
                synthetic = Form('list', '(', (
                    Form('atom', 'defn', (), f.line, f.col),
                    method.name_form, *f.children[3:]),
                    f.line, f.col, (), f.end_line, f.end_col)
                g['clojure_semantic_analysis'][method_qual] = analyze_declaration(
                    synthetic, namespace=ns[0] if ns else '',
                    phase='runtime_candidate', core_short_circuit=short_circuit)
                pending_methods.append((path, implementation, method.name_form))
                # The signature has fn-like lexical scope. Its body is
                # independent of the dispatch expression (which is evaluated
                # while installing the method, not when the method is called).
                function_form = Form('list', '(', (
                    Form('internal_function_scope', 'fn*', (), f.line, f.col),
                    *f.children[3:]), f.line, f.col)
                if referred:
                    sites, suppressions = referred_call_sites(
                        (function_form,), frozenset(referred),
                        core_shadowed=core_shadowed, macro_aliases=macro_aliases)
                    for site in sites:
                        pending_referred_calls.append((path, implementation,
                            site.local_name, referred[site.local_name], site.head,
                            'runtime_candidate', site.argument_count))
                    for suppressed in suppressions:
                        g['lexical_scope_suppression_count'] += 1
                        if len(g['lexical_scope_suppression_examples']) < 40:
                            g['lexical_scope_suppression_examples'].append({
                                'name': suppressed.name, 'reason': suppressed.reason,
                                'evidence': ev(suppressed.site.line,suppressed.site.col)})
                # defmethod's dispatch *value* expression is evaluated when
                # the method is registered, not when its fn body is called.
                # Keep separate phases and identical source anchoring.
                for scan_forms, phase in (
                    ((function_form,), 'runtime_candidate'),
                    ((method.dispatch_form,), 'namespace_initialization_candidate')):
                    if referred and phase != 'runtime_candidate':
                        sites, suppressions = referred_call_sites(
                            scan_forms, frozenset(referred),
                            core_shadowed=core_shadowed, macro_aliases=macro_aliases)
                        for site in sites:
                            pending_referred_calls.append((path, implementation,
                                site.local_name, referred[site.local_name],site.head,
                                phase, site.argument_count))
                        for suppressed in suppressions:
                            g['lexical_scope_suppression_count'] += 1
                            if len(g['lexical_scope_suppression_examples']) < 40:
                                g['lexical_scope_suppression_examples'].append({
                                    'name': suppressed.name, 'reason': suppressed.reason,
                                    'evidence':ev(suppressed.site.line,suppressed.site.col)})
                    sites, suppressions = referred_call_sites(scan_forms,
                        frozenset(), include_qualified=True,
                        core_shadowed=core_shadowed, macro_aliases=macro_aliases)
                    for site in sites:
                        prefix, member = site.head.value.split('/', 1)
                        if not member:continue
                        if prefix in data['alias_guards']:
                            pending_alias_issues.append((path,site.head,data['alias_guards'][prefix]))
                        elif prefix in aliases or '.' in prefix or prefix in namespace_names:
                            pending_calls.append((path,implementation,aliases.get(prefix,prefix),
                                member,site.head,phase,site.argument_count))
                    for suppressed in suppressions:
                        g['qualified_scope_suppression_count'] += 1
                        if len(g['qualified_scope_suppression_examples']) < 40:
                            g['qualified_scope_suppression_examples'].append({
                                'name':suppressed.name,'reason':suppressed.reason,
                                'evidence':ev(suppressed.site.line,suppressed.site.col)})
                continue
            name,kind,declpart=parsed
            qual=(ns[0]+'/' if ns else '')+name
            declaration=node(ns_node or fnode,name,qual,kind,declpart)
            if kind in {'function','macro','multimethod','variable','protocol','record','type','interface'}:
                visibility,visibility_reason=_callable_visibility(f)
                declaration['visibility']=visibility
                declaration['visibility_evidence']=visibility_reason
            if ns:
                def_index.setdefault((ns[0],name),[]).append(declaration)
            if kind in {'record','type'}:
                direct=parse_direct_implementations(f,forms,referred,
                    platform=_source_platform(path))
                if direct:
                    publish_protocol_sites(direct,declaration)
            if kind in {'protocol','type','record','interface'}:
                members,reason=_literal_declaration_members(f,kind)
                if reason:
                    if len(g['declaration_limitations'])<MAX_UNRESOLVED_EXAMPLES:
                        g['declaration_limitations'].append({
                            'reason':reason,'declaration':qual,
                            'evidence':ev(f.line,f.col)})
                for member_name,member_kind,part in members:
                    member=node(declaration,member_name,qual+'/'+member_name,
                                member_kind,part)
                    member['semantic_resolution']='declaration_syntax_only_dispatch_unresolved'
                    from .clojure_type_hints import literal_type_tag
                    if (tag:=literal_type_tag(part)) is not None:
                        member['literal_type_hint']=tag
                    if member_kind=='interface_method_declaration':
                        sig=next((b for b in f.children[2:] if b.kind=='list' and
                          b.children[0] is part),None)
                        if sig is not None:
                            member['declared_fixed_arities']=[len(sig.children[1].children)]
                            member['interface_receiver_parameter_excluded']=True
                            member['arity_evidence']='literal_jvm_definterface_method_signature'
                        member['visibility']=declaration.get('visibility','unknown')
                        member['semantic_resolution']='jvm_interface_method_syntax_not_dispatch'
                    if member_kind=='protocol_method_declaration':
                        # defprotocol source syntax declares a polymorphic
                        # function Var, but does not select any implementing
                        # type. Keep the declared member's stable source ID.
                        method_form=next((body for body in f.children[2:]
                            if body.kind=='list' and body.children and
                               body.children[0].value==member_name),None)
                        if method_form is not None:
                            member['declared_fixed_arities']=sorted({
                                len(args.children) for args in method_form.children[1:]
                                if args.kind=='vector'})
                            member['declared_variadic_min_arity']=None
                            member['arity_evidence']='literal_protocol_method_signature_only'
                        member['visibility']=declaration.get('visibility','unknown')
                        member['protocol_generated_var_candidate']=(
                            ((ns[0]+'/' if ns else '')+member_name))
                        if ns:
                            def_index.setdefault((ns[0],member_name),[]).append(member)
            if kind in ('function','macro'):
                from .clojure_scope import _fn_arities
                pieces=list(f.children[2:])
                if pieces and pieces[0].kind=='string':pieces.pop(0)
                if pieces and pieces[0].kind=='map':pieces.pop(0)
                if pieces and pieces[0].kind=='list' and pieces[-1].kind=='map':pieces.pop()
                clauses=_fn_arities(tuple(pieces))
                if clauses is not None:
                    fixed=[];variadic=None
                    for params,_ in clauses:
                        symbols=[part.value for part in params.children]
                        if '&' in symbols:
                            n=symbols.index('&')
                            variadic=n
                        else:
                            fixed.append(len(symbols))
                    declaration['declared_fixed_arities']=sorted(fixed)
                    declaration['declared_variadic_min_arity']=variadic
                    declaration['arity_evidence']='literal_defn_signature_not_compiler_confirmed'
                from .clojure_semantics import analyze_declaration
                # Only literal defn bodies.  Macros are analyzed as source
                # declarations, not executed or falsely expanded.
                g['clojure_semantic_analysis'][qual] = analyze_declaration(
                    f, namespace=ns[0] if ns else '',
                    phase='macro_expansion' if kind=='macro' else 'runtime',
                    core_short_circuit=short_circuit)
            if kind == 'variable':
                from .clojure_var_initializer import parse_variable_initializer
                init = parse_variable_initializer(f)
                declaration['variable_initializer_evidence'] = init.evidence()
                if init.status == 'unsupported':
                    g['declaration_limitations'].append({
                        'reason': init.reason,
                        'declaration': qual,
                        'evidence': ev(f.line,f.col)})
            if kind in ('function','macro','multimethod','variable'):
                phase=('macro_expansion' if kind=='macro' else
                       'namespace_initialization_candidate' if kind=='variable' else
                       'runtime_candidate')
                # Clojure unqualified :refer calls are searched only at true
                # function-call heads with tracked lexical bindings.  A local
                # param/let/fn binding shadows the imported name, not a cross-
                # file dependency. Unsupported scopes are recorded, not linked.
                if referred:
                    body=(list(init.forms) if kind=='variable' else
                          list(f.children[2:]))
                    if kind in ('function','macro'):
                        if body and body[0].kind=='string':body.pop(0)
                        if body and body[0].kind=='map':body.pop(0)
                        if body and body[0].kind=='list' and body[-1].kind=='map':body.pop()
                        # Synthetic function environment, not a project macro
                        # invocation: namespace exclusions cannot shadow it.
                        fn_head=Form('internal_function_scope','fn*',(),f.line,f.col)
                        fn_form=Form('list','(',tuple([fn_head,*body]),f.line,f.col)
                        sites,suppressions=referred_call_sites((fn_form,),frozenset(referred),name,core_shadowed=core_shadowed,macro_aliases=macro_aliases,
                            implicit_bindings=frozenset({'&form','&env'}) if kind=='macro' else frozenset())
                    else:
                        sites,suppressions=referred_call_sites(tuple(body),frozenset(referred),name,core_shadowed=core_shadowed,macro_aliases=macro_aliases)
                    for site in sites:
                        pending_referred_calls.append((path,declaration,site.local_name,
                                                       referred[site.local_name],site.head,phase,site.argument_count))
                    for suppressed in suppressions:
                        g['lexical_scope_suppression_count']+=1
                        if len(g['lexical_scope_suppression_examples'])<40:
                            g['lexical_scope_suppression_examples'].append({
                                'name':suppressed.name,'reason':suppressed.reason,
                                'evidence':ev(suppressed.site.line,suppressed.site.col)})
                # Use the SAME bounded lexical scan for qualified call heads.
                # Old raw recursive traversal misinterpreted unverified macros
                # (for/doseq/etc.) as executable call sites.
                qualified_body=(list(init.forms) if kind=='variable' else
                                list(f.children[2:]))
                if kind in ('function','macro'):
                    if qualified_body and qualified_body[0].kind=='string':
                        qualified_body.pop(0)
                    if qualified_body and qualified_body[0].kind=='map':
                        qualified_body.pop(0)
                    if qualified_body and qualified_body[0].kind=='list' and qualified_body[-1].kind=='map':qualified_body.pop()
                    fn_head=Form('internal_function_scope','fn*',(),f.line,f.col)
                    qualified_forms=(Form('list','(',tuple([fn_head,*qualified_body]),f.line,f.col),)
                else:
                    qualified_forms=tuple(qualified_body)
                qualified_sites,qualified_suppressions=referred_call_sites(
                    qualified_forms, frozenset(), name, include_qualified=True,core_shadowed=core_shadowed,macro_aliases=macro_aliases,
                    implicit_bindings=frozenset({'&form','&env'}) if kind=='macro' else frozenset())
                for site in qualified_sites:
                    called=site.head
                    prefix,member=called.value.split('/',1)
                    if not member:
                        continue
                    if prefix in data['alias_guards']:
                        pending_alias_issues.append((path,called,data['alias_guards'][prefix]))
                        continue
                    if prefix in aliases or '.' in prefix or prefix in namespace_names:
                        target_ns=aliases.get(prefix,prefix)
                        pending_calls.append((path,declaration,target_ns,member,called,phase,site.argument_count))
                for suppressed in qualified_suppressions:
                    g['qualified_scope_suppression_count']+=1
                    if len(g['qualified_scope_suppression_examples'])<40:
                        g['qualified_scope_suppression_examples'].append({
                            'name':suppressed.name,'reason':suppressed.reason,
                            'evidence':ev(suppressed.site.line,suppressed.site.col)})
                for item in f.children[2:]:
                    for special,form_name in _walk_special_forms(item):
                        special_seen+=1
                        if special_seen > MAX_SYNTAX_NODES:
                            continue
                        location=f'{qual}/syntax:{form_name}@{special.line}:{special.col}'
                        part=Form('atom',form_name,(),special.line,special.col)
                        syntax=node(declaration,form_name,location,'clojure_syntax_form',part,relation='contains_syntax')
                        syntax['analysis_phase']=phase
                        syntax['semantic_resolution']='lexical_structure_only_no_control_flow_claim'
                if special_seen > MAX_SYNTAX_NODES:
                    warnings.append({'path':path,'reason':'clojure_syntax_nodes_truncated',
                                     'details':f'limit={MAX_SYNTAX_NODES}'})
        # A shared .cljc source may contain declarations in #? branches which
        # are not top-level definitions in the unprojected reader tree.  Those
        # definitions were present only in platform_projections metadata and
        # were INVISIBLE in Inside File.  Represent their syntax in the graph,
        # but do not add them to ns_index/def_index or resolve call targets:
        # selecting a platform and executing macros/build config is outside
        # this source-only pass.  Keep the JVM and CLJS symbols distinct.
        if _source_platform(path) == 'shared':
            already_visible = {
                (parsed[0], parsed[1], parsed[2].line, parsed[2].col)
                for form in forms if (parsed := _definition(form))
            }
            already_visible.update(
                (item['name'], item['kind'], item['line'], item['column'])
                for item in g['nodes'] if item.get('kind') in
                {'multimethod_method','protocol_implementation_method'}
                and item.get('confidence')=='parsed_structural_subset')
            projected_seen = set()
            projected_count = 0
            projected_omitted = 0
            for target_platform in ('clj', 'cljs'):
                projection = data['platform_projections'].get(target_platform, {})
                if projection.get('status') != 'candidate_not_semantically_verified':
                    continue
                projected_ns = projection.get('namespace')
                platform_forms = data['_platform_projected_forms'].get(target_platform, ())
                projected_referred = _explicit_refers(list(platform_forms), target_platform)[0]
                for projected_form in platform_forms:
                    parsed = _definition(projected_form)
                    projected_method = None
                    projected_extensions = None
                    if parsed is None:
                        projected_method = parse_method_site(
                            projected_form, platform_forms, projected_referred,
                            platform=target_platform)
                        projected_extensions = parse_extension_methods(
                            projected_form, platform_forms, projected_referred,
                            platform=target_platform)
                    projected_direct = parse_direct_implementations(projected_form,
                        platform_forms,projected_referred,platform=target_platform)
                    projected_literal_extend = parse_literal_extend(projected_form,
                        platform_forms,projected_referred,platform=target_platform)
                    projected_nested = tuple(walk_reify_sites(projected_form,
                        platform_forms,projected_referred,platform=target_platform))
                    definitions = ([parsed] if parsed is not None else
                        [(projected_method.name_form.value,'multimethod_method',
                          projected_method.name_form)] if projected_method is not None else
                        [(site.method.children[0].value,'protocol_implementation_method',
                          site.method.children[0]) for site in (projected_extensions or ())])
                    definitions.extend((site.name_form.value,'protocol_implementation_method',
                        site.name_form) for site in (* (projected_direct or ()),
                          *(projected_literal_extend or ()), *projected_nested))
                    for name, kind, part in definitions:
                        site = (name, kind, part.line, part.col)
                        if site in already_visible:
                            continue  # ordinary shared declaration already exists
                        key = (target_platform, *site)
                        if key in projected_seen:
                            continue
                        projected_seen.add(key)
                        if projected_count >= MAX_SYNTAX_NODES:
                            projected_omitted += 1
                            continue
                        qualified = (projected_ns + '/' if projected_ns else '') + name
                        # The platform suffix is identity, not a source name.  A
                        # CLJ declaration never aliases the corresponding CLJS one.
                        candidate = {
                            'id': _node_id(path, qualified + '#platform:' + target_platform,
                                           part.line, part.col),
                            'path': path, 'name': name, 'qualname': qualified,
                            'kind': kind, 'line': part.line, 'column': part.col,
                            'parent': (ns_node['id'] if ns_node is not None and
                                       ns is not None and ns[0] == projected_ns else fnode['id']),
                            'confidence': 'projected_platform_candidate',
                            'source_platform': target_platform,
                            'platform_binding_status': 'source_branch_only_not_resolved',
                        }
                        if part.end_line is not None and part.end_col is not None:
                            candidate['end_line'] = part.end_line
                            candidate['end_column'] = part.end_col
                        if kind=='protocol_implementation_method' and projected_extensions:
                            extension=next((site for site in projected_extensions
                                if site.method.children[0] is part),None)
                            if extension is not None:
                                candidate['protocol_source_name']=extension.protocol.value
                                candidate['type_source_name']=extension.type_form.value
                                candidate['semantic_resolution']=(
                                    'projected_protocol_extension_source_only_not_runtime_dispatch')
                                candidate['runtime_dispatch_confirmed']=False
                                candidate['type_binding_confirmed']=False
                        if projected_method is not None:
                            candidate['semantic_resolution'] = (
                                'projected_multimethod_registration_not_compiler_verified')
                            candidate['dispatch_evidence'] = {
                                'expression_kind':projected_method.dispatch_form.kind,
                                'source': ev(projected_method.dispatch_form.line,
                                    projected_method.dispatch_form.col,
                                    end_line=projected_method.dispatch_form.end_line,
                                    end_column=projected_method.dispatch_form.end_col),
                                'runtime_dispatch_confirmed':False,
                                'macroexpansion_performed':False}
                        g['nodes'].append(candidate)
                        g['edges'].append({
                            'source': candidate['parent'], 'target': candidate['id'],
                            'source_path': path, 'target_path': path,
                            'relation': 'declares',
                            'resolution': 'cljc_platform_projection_candidate_not_compiler_binding',
                            'analyzer': 'clojure-reader-conditional-projection',
                            'source_platform': target_platform,
                            'evidence': ev(part.line, part.col, end_line=part.end_line,
                                           end_column=part.end_col),
                        })
                        projected_count += 1
            g['platform_projection_graph_nodes'] = projected_count
            if projected_omitted:
                g['platform_projection_graph_omitted'] = projected_omitted
                warnings.append({'path': path, 'reason': 'cljc_projected_graph_node_limit',
                                 'details': f'limit={MAX_SYNTAX_NODES}; omitted={projected_omitted}'})
        g['_evidence']=ev
    # Read-only project Var VALUE references (not only invocation heads).
    # A Var can be passed as a value, returned, stored, or referenced by (var x).
    # Resolve only a unique literal source declaration with the same exact
    # source/classpath guards as call-head references. Macro arguments remain
    # candidates: no project macro is expanded or executed by this pass.
    from .clojure_scope import referred_call_sites as _scoped_value_sites
    pending_value_references=[]
    for path,data in sorted(files.items()):
        ns=data['namespace'][0]
        if not ns or len(ns_index.get(ns[0],[]))!=1:
            continue
        namespace=ns[0]
        own_names=frozenset(name for (decl_ns,name),nodes in def_index.items()
            if decl_ns==namespace and any(n['path']==path for n in nodes))
        foreign_refers=data['referred_names'][0]
        names=own_names | frozenset(foreign_refers)
        if not names and not data['namespace'][1]:
            continue
        from .clojure_core_context import shadowed_core_names,core_macro_aliases
        shadowed=shadowed_core_names(data['forms'],foreign_refers)
        aliases=core_macro_aliases(data['forms'])
        for form in data['forms']:
            definition=_definition(form)
            if not definition or definition[1] not in {'function','macro','variable','multimethod'}:
                continue
            caller_name,kind,_=definition
            owners=[n for n in graphs[path]['nodes'] if
                    n.get('qualname')==namespace+'/'+caller_name and n['kind']==kind]
            if len(owners)!=1:
                continue
            if kind == 'variable':
                from .clojure_var_initializer import parse_variable_initializer
                parts=list(parse_variable_initializer(form).forms)
            else:
                parts=list(form.children[2:])
            if kind in ('function','macro'):
                if parts and parts[0].kind=='string':parts.pop(0)
                if parts and parts[0].kind=='map':parts.pop(0)
                if parts and parts[0].kind=='list' and parts[-1].kind=='map':parts.pop()
                marker=Form('internal_function_scope','fn*',(),form.line,form.col)
                body=(Form('list','(',tuple([marker,*parts]),form.line,form.col),)
            else:
                body=tuple(parts)
            sites,ignored=_scoped_value_sites(body,names,None,
                include_qualified=True,include_values=True,include_call_heads=False,
                core_shadowed=shadowed,macro_aliases=aliases,
                implicit_bindings=frozenset({'&form','&env'}) if kind=='macro' else frozenset())
            graphs[path]['value_reference_scope_suppression_count']= (
                graphs[path].get('value_reference_scope_suppression_count',0)+len(ignored))
            if ignored:
                graphs[path].setdefault('value_reference_scope_suppression_examples',[])
                for item in ignored[:max(0,20-len(graphs[path]['value_reference_scope_suppression_examples']))]:
                    graphs[path]['value_reference_scope_suppression_examples'].append({
                        'symbol':item.name,'reason':item.reason,
                        'evidence':graphs[path]['_evidence'](item.site.line,item.site.col)})
            if len(sites)>2000:
                warnings.append({'path':path,'reason':'clojure_var_value_site_limit',
                                 'details':f'{caller_name}: {len(sites)} sites, first 2000 only'})
            for site in sites[:2000]:
                pending_value_references.append((path,owners[0],site.head,
                    ('macro_expansion' if kind=='macro' else
                     'namespace_initialization_candidate' if kind=='variable' else
                     'runtime_candidate'), site.reference_form))

    # Call arity is evidence about the literal source expression and the
    # indexed defn/defmacro signature ONLY. It is not proof of the current Var
    # root, runtime callability, macro expansion, or JVM/CLJS dispatch.
    # A wrong-arity expression still names the same Var; never remove its
    # candidate edge merely because the source signature does not match.
    def literal_call_arity_evidence(argument_count, declaration, source_platform):
        result={'source_argument_count':argument_count,
                'call_binding_confirmed':False,
                'source_call_arity_provenance':'literal_list_element_count_no_macroexpansion',
                'declaration_call_arity_provenance':'literal_defn_signature_not_compiler_verified',
                'source_call_arity_semantic_warning':'runtime_var_root_and_macro_expansion_not_verified'}
        if argument_count is None:
            result['declaration_arity_comparison']='unavailable_not_direct_literal_call'
            return result
        if source_platform=='shared' or _source_platform(declaration['path'])=='shared':
            result['declaration_arity_comparison']='unavailable_shared_platform_not_compiler_selected'
            return result
        if (declaration.get('kind') not in {'function','macro'} or
                declaration.get('arity_evidence')!='literal_defn_signature_not_compiler_confirmed'):
            result['declaration_arity_comparison']='unavailable_no_validated_literal_function_signature'
            return result
        fixed=declaration.get('declared_fixed_arities')
        minimum=declaration.get('declared_variadic_min_arity')
        if not isinstance(fixed,list) or any(type(n)!=int or n<0 for n in fixed) or (
                minimum is not None and (type(minimum)!=int or minimum<0)):
            result['declaration_arity_comparison']='unavailable_malformed_signature_evidence'
            return result
        result['target_literal_fixed_arities']=list(fixed)
        result['target_literal_variadic_min_arity']=minimum
        if argument_count in fixed:
            result['declaration_arity_comparison']='matches_literal_fixed_arity'
        elif minimum is not None and argument_count>=minimum:
            result['declaration_arity_comparison']='matches_literal_variadic_arity'
        else:
            result['declaration_arity_comparison']='not_declared_by_literal_source_signature'
        return result

    def initializer_site_context(phase, owner):
        if phase != 'namespace_initialization_candidate':
            return {}
        init = owner.get('variable_initializer_evidence', {})
        return {
            'source_expression_role':'literal_var_initializer_expression',
            'source_initializer_status':init.get('status','unverified'),
            'initializer_execution_confirmed':False,
            'defonce_initializer_may_be_skipped':(
                init.get('status')=='conditional_defonce_initializer_candidate'),
            'nested_deferred_expression_may_not_execute_at_initialization':True,
        }

    # Resolve *declaration identities* for same-namespace call heads.  This
    # intentionally does not claim dispatch, a stable Var root, macro
    # expansion, or runtime reachability. A lexical local/parameter shadows
    # a Var name and must not create an edge.
    for path,data in sorted(files.items()):
        ns=data['namespace'][0]
        if not ns or len(ns_index.get(ns[0],[]))!=1:
            continue
        ns_name=ns[0]
        local_names=frozenset(name for (namespace,name),nodes in def_index.items()
                              if namespace==ns_name and any(
                                  node['path']==path and
                                  node['kind'] in {'function','macro','multimethod'}
                                  for node in nodes))
        if not local_names:
            continue
        g=graphs[path]
        from .clojure_core_context import shadowed_core_names, core_macro_aliases
        core_shadowed=shadowed_core_names(data['forms'],data['referred_names'][0])
        macro_aliases=core_macro_aliases(data['forms'])
        seen_local=set()
        for form in data['forms']:
            recognized=_definition(form)
            if not recognized or recognized[1] not in {'function','macro','variable'}:
                continue
            caller_name,kind,_=recognized
            caller_nodes=[n for n in g['nodes'] if n.get('qualname')==ns_name+'/'+caller_name
                          and n['kind']==kind]
            if len(caller_nodes)!=1:
                continue
            caller=caller_nodes[0]
            if kind == 'variable':
                from .clojure_var_initializer import parse_variable_initializer
                body=parse_variable_initializer(form).forms
                scoped_forms=body
            else:
                body=list(form.children[2:])
                if body and body[0].kind=='string':body.pop(0)
                if body and body[0].kind=='map':body.pop(0)
                if body and body[0].kind=='list' and body[-1].kind=='map':body.pop()
                marker=Form('internal_function_scope','fn*',(),form.line,form.col)
                scoped=Form('list','(',tuple([marker,*body]),form.line,form.col)
                scoped_forms=(scoped,)
            sites,ignored=referred_call_sites(scoped_forms,local_names,caller_name,
                core_shadowed=core_shadowed,macro_aliases=macro_aliases,
                implicit_bindings=frozenset({'&form','&env'}) if kind=='macro' else frozenset())
            for site in sites:
                name=site.local_name
                matches=[n for n in def_index.get((ns_name,name),[])
                         if n['path']==path and
                         n['kind'] in {'function','macro','multimethod','protocol_method_declaration'}]
                if len(matches)!=1:
                    continue  # cannot prove a unique Var declaration
                target=matches[0]
                sig=(caller['id'],target['id'],site.head.line,site.head.col)
                if sig in seen_local:
                    continue
                seen_local.add(sig)
                g['edges'].append({'source':caller['id'],'target':target['id'],
                    'source_path':path,'target_path':path,
                    'relation':'clojure_local_var_reference_candidate',
                    'resolution':'clojure_local_declaration_identity_not_runtime_call',
                    'analyzer':'clojure-scoped-static-var-index',
                    'source_platform':_source_platform(path),
                    'analysis_phase':('macro_expansion' if kind=='macro' else
                        'namespace_initialization_candidate' if kind=='variable' else
                        'runtime_candidate'),
                    'target_declaration_kind':target['kind'],
                    **initializer_site_context(
                        'namespace_initialization_candidate' if kind=='variable' else '', caller),
                    **literal_call_arity_evidence(site.argument_count,target,_source_platform(path)),
                    'evidence':g['_evidence'](site.head.line,site.head.col,
                                            end_line=site.head.end_line,
                                            end_column=site.head.end_col)})
            if ignored:
                g.setdefault('local_var_scope_suppression_count',0)
                g['local_var_scope_suppression_count']+=len(ignored)
    def unresolved(path, kind, symbol, line, col, reason):
        g=graphs[path]
        g['unresolved_reference_count']+=1
        if len(g['unresolved_references'])<MAX_UNRESOLVED_EXAMPLES:
            g['unresolved_references'].append({
                'kind':kind,'symbol':symbol[:180],'reason':reason,
                'source_platform':_source_platform(path),
                'evidence':g['_evidence'](line,col)})

    def namespace_target(path, name):
        available=ns_index.get(name,[])
        context=source_contexts[path]
        candidate_paths,context_reason=select_namespace_paths(context,name,_source_platform(path),
            [node['path'] for node in available])
        if context_reason:return None,context_reason
        selected=set(candidate_paths)
        matches=_select_namespace_targets([node for node in available if node['path'] in selected],path)
        if matches:
            return matches[0],None
        if not available:
            return None,'namespace_external_or_not_indexed'
        platforms={_source_platform(n['path']) for n in available}
        if _source_platform(path)=='shared' and platforms & {'clj','cljs'}:
            return None,'shared_source_platform_not_selected'
        if len(available)>1:
            return None,'ambiguous_or_incompatible_platform_namespace'
        return None,'platform_incompatible_namespace'

    # defmethod installs a method for a MultiFn; it never declares a new Var.
    # A source-anchored installation edge does not imply that a runtime
    # dispatch value will select the implementation (hierarchies are dynamic).
    for path, owner, site in pending_methods:
        symbol=site.value
        current_ns=files[path]['namespace'][0]
        if not current_ns:
            unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                       'namespace_unavailable_for_multimethod')
            continue
        namespace=current_ns[0]
        member=symbol
        if '/' in symbol:
            prefix,member=symbol.split('/',1)
            if prefix in files[path]['alias_guards']:
                unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                           files[path]['alias_guards'][prefix]);continue
            aliases=files[path]['namespace'][1]
            if prefix not in aliases and prefix != namespace and prefix not in namespace_names:
                unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                           'unverified_multimethod_namespace_qualifier');continue
            namespace=aliases.get(prefix,prefix)
        elif not def_index.get((namespace,symbol)):
            binding=files[path]['referred_names'][0].get(symbol)
            if binding is None:
                unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                           'no_unique_local_or_explicit_referred_multimethod');continue
            namespace,member=binding
        target_ns, reason=namespace_target(path,namespace)
        if target_ns is None:
            unresolved(path,'multimethod_registration',symbol,site.line,site.col,reason)
            continue
        matches=[item for item in def_index.get((namespace,member),[])
                 if item['path']==target_ns['path'] and item['kind']=='multimethod']
        if len(matches)!=1:
            unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                'multimethod_not_declared_or_ambiguous')
            continue
        target=matches[0]
        if target['path']!=path and target.get('visibility')!='public':
            unresolved(path,'multimethod_registration',symbol,site.line,site.col,
                'multimethod_not_public_or_visibility_unknown')
            continue
        graphs[path]['edges'].append({
            'source': owner['id'], 'target':target['id'],
            'source_path':path,'target_path':target['path'],
            'relation':'clojure_multimethod_registration_candidate',
            'resolution':'source_only_multimethod_registration_candidate',
            'analyzer':'clojure-static-multimethod',
            'source_platform':_source_platform(path),
            'target_platform':_source_platform(target['path']),
            'runtime_dispatch_confirmed':False,
            'call_binding_confirmed':False,
            'evidence':graphs[path]['_evidence'](site.line,site.col,
                end_line=site.end_line,end_column=site.end_col)})

    # Protocol extension methods are source declarations. Resolve only a
    # unique *declared* protocol member; never infer Java class identity,
    # runtime extension installation or polymorphic invocation selection.
    for path, owner, protocol_form, member_form in pending_extensions:
        spelling=protocol_form.value
        # Clojure JVM definterface uses Java class names (ns.Interface),
        # not a protocol Var. Keep class dispatch separate from protocols.
        # Cross-file lookup is permitted ONLY for a declared project namespace,
        # never for a guessed JAR class or a shared CLJC compiler target.
        if _source_platform(path)=='clj' and '/' not in spelling:
            current=files[path]['namespace'][0]
            iface_ns=None;iface_name=spelling
            if '.' in spelling:
                iface_ns,iface_name=spelling.rsplit('.',1)
            elif current:
                iface_ns=current[0]
            if iface_ns:
                interface_defs=[d for d in def_index.get((iface_ns,iface_name),[])
                    if d['kind']=='interface' and _source_platform(d['path'])=='clj']
                if len(interface_defs)==1 and (
                        iface_ns!=current[0] or interface_defs[0]['path']==path):
                    iface=interface_defs[0]
                    iface_target,reason=namespace_target(path,iface_ns)
                    if iface_target and iface_target['path']==iface['path']:
                        members=[n for n in graphs[iface['path']]['nodes']
                            if n['kind']=='interface_method_declaration' and
                            n['name']==member_form.value and n['parent']==iface['id']]
                        provided=owner.get('declared_fixed_arities',[])
                        # Source method params contain receiver `this`, while
                        # definterface declares only Java method parameters.
                        matches=[m for m in members if
                            owner.get('arity_evidence')=='literal_implementation_source_only' and
                            provided and not owner.get('declared_variadic_min_arity') and
                            set(provided)=={a+1 for a in m.get('declared_fixed_arities',[])}]
                        if len(matches)==1:
                            graphs[path]['edges'].append({
                                'source':owner['id'],'target':matches[0]['id'],
                                'source_path':path,'target_path':iface['path'],
                                'relation':'clojure_jvm_interface_implementation_candidate',
                                'resolution':'unique_project_definterface_source_method_candidate',
                                'analyzer':'clojure-static-jvm-interface-source',
                                'source_platform':'clj','target_platform':'clj',
                                'runtime_dispatch_confirmed':False,
                                'type_binding_confirmed':False,
                                'compiler_binding_confirmed':False,
                                'evidence':graphs[path]['_evidence'](member_form.line,member_form.col,
                                    end_line=member_form.end_line,end_column=member_form.end_col)})
                            continue
                        unresolved(path,'jvm_interface_implementation',spelling,
                            protocol_form.line,protocol_form.col,
                            'interface_method_signature_missing_or_mismatched')
                        continue
        
        current_ns=files[path]['namespace'][0]
        if not current_ns:
            unresolved(path,'protocol_extension',spelling,protocol_form.line,
                protocol_form.col,'protocol_extension_namespace_unavailable')
            continue
        namespace=current_ns[0]
        protocol_name=spelling
        if '/' in spelling:
            prefix, protocol_name=spelling.split('/',1)
            if prefix in files[path]['alias_guards']:
                unresolved(path,'protocol_extension',spelling,protocol_form.line,
                    protocol_form.col,files[path]['alias_guards'][prefix]);continue
            aliases=files[path]['namespace'][1]
            if prefix not in aliases and prefix!=namespace and prefix not in namespace_names:
                unresolved(path,'protocol_extension',spelling,protocol_form.line,
                    protocol_form.col,'protocol_qualifier_not_indexed');continue
            namespace=aliases.get(prefix,prefix)
        elif not def_index.get((namespace,spelling)):
            referred=files[path]['referred_names'][0].get(spelling)
            if referred is None:
                unresolved(path,'protocol_extension',spelling,protocol_form.line,
                    protocol_form.col,'protocol_declaration_or_explicit_refer_unknown')
                continue
            namespace,protocol_name=referred
        ns_target,reason=namespace_target(path,namespace)
        if ns_target is None:
            unresolved(path,'protocol_extension',spelling,protocol_form.line,
                protocol_form.col,reason)
            continue
        protocols=[p for p in def_index.get((namespace,protocol_name),[])
                   if p['path']==ns_target['path'] and p['kind']=='protocol']
        if len(protocols)!=1:
            unresolved(path,'protocol_extension',spelling,protocol_form.line,
                protocol_form.col,'protocol_declaration_missing_or_ambiguous')
            continue
        parent=protocols[0]
        if parent['path']!=path and parent.get('visibility')!='public':
            unresolved(path,'protocol_extension',spelling,protocol_form.line,
                protocol_form.col,'protocol_visibility_not_proven_public')
            continue
        members=[n for n in graphs[parent['path']]['nodes']
                 if n['kind']=='protocol_method_declaration' and
                 n['name']==member_form.value and n['parent']==parent['id']]
        if len(members)!=1:
            unresolved(path,'protocol_extension',member_form.value,
                member_form.line,member_form.col,
                'protocol_method_not_declared_or_signature_unknown')
            continue
        if owner.get('arity_evidence')=='literal_implementation_source_only':
            declared=members[0].get('declared_fixed_arities')
            supplied=owner.get('declared_fixed_arities',[])
            if (declared is None or not supplied or
                    not set(supplied).issubset(set(declared)) or
                    owner.get('declared_variadic_min_arity') is not None):
                unresolved(path,'protocol_extension',member_form.value,
                    member_form.line,member_form.col,
                    'protocol_implementation_arity_not_in_literal_protocol_signature')
                continue
        graphs[path]['edges'].append({
            'source':owner['id'],'target':members[0]['id'],
            'source_path':path,'target_path':parent['path'],
            'relation':'clojure_protocol_implementation_candidate',
            'resolution':'unique_project_protocol_method_syntax_candidate',
            'analyzer':'clojure-static-protocol-extension',
            'source_platform':_source_platform(path),
            'target_platform':_source_platform(parent['path']),
            'runtime_dispatch_confirmed':False,'type_binding_confirmed':False,
            'call_binding_confirmed':False,
            'evidence':graphs[path]['_evidence'](member_form.line,member_form.col,
                end_line=member_form.end_line,end_column=member_form.end_col)})

    for path,part,reason in pending_alias_issues:
        unresolved(path,'qualified_call',part.value,part.line,part.col,reason)
    for path,data in files.items():
        for issue in data.get('unresolved_libspecs', []):
            unresolved(path,'unverified_require_libspec',issue['symbol'],
                       issue['line'],issue['column'],issue['reason'])

    for path,owner,name,line,col in pending_requires:
        target,reason=namespace_target(path,name)
        if target is None:
            unresolved(path,'literal_require',name,line,col,reason)
            continue
        graphs[path]['edges'].append({'source':owner['id'],'target':target['id'],
           'source_path':path,'target_path':target['path'],'relation':'clojure_require_candidate',
           'resolution':'clojure_literal_ns_candidate','analyzer':'clojure-form-reader',
           'source_platform':_source_platform(path),'target_platform':_source_platform(target['path']),
           'evidence':graphs[path]['_evidence'](line,col)})
    seen=set()
    for path,owner,namespace,member,part,phase,argument_count in pending_calls:
        ns_target,reason=namespace_target(path,namespace)
        if ns_target is None:
            unresolved(path,'qualified_call',namespace+'/'+member,part.line,part.col,reason)
            continue
        target_path=ns_target['path']
        matches=[n for n in def_index.get((namespace,member),[])
                 if n['path']==target_path and n['kind'] in {'function','macro','multimethod','protocol_method_declaration'}]
        if len(matches)!=1:
            unresolved(path,'qualified_call',namespace+'/'+member,part.line,part.col,
                       'target_not_callable_or_not_indexed' if not matches else 'ambiguous_callable_definitions')
            continue
        target=matches[0]
        if target['kind']=='protocol_method_declaration':
            # The protocol function Var is identifiable in source, but which
            # implementation handles a call is intentionally NOT resolved.
            unresolved(path,'qualified_call',namespace+'/'+member,
                part.line,part.col,'protocol_runtime_dispatch_unresolved')
        if target_path!=path and target.get('visibility')!='public':
            unresolved(path,'qualified_call',namespace+'/'+member,part.line,part.col,
                       'private_var_not_public_cross_namespace' if target.get('visibility')=='private'
                       else 'target_var_visibility_not_statistically_verified')
            continue
        sig=(owner['id'],target['id'],part.line,part.col,phase)
        if sig in seen:continue
        seen.add(sig)
        if phase=='macro_expansion':
            relation='clojure_macro_phase_candidate'
        elif target['kind']=='macro':
            relation='clojure_macro_invocation_candidate'
        elif target['kind']=='protocol_method_declaration':
            relation='clojure_protocol_method_call_candidate'
        else:
            relation='clojure_call_candidate'
        graphs[path]['edges'].append({'source':owner['id'],'target':target['id'],
           'source_path':path,'target_path':target['path'],'relation':relation,
           'resolution':'clojure_qualified_name_candidate','analyzer':'clojure-form-reader',
           'source_platform':_source_platform(path),'target_platform':_source_platform(target['path']),
           'analysis_phase':phase,'target_declaration_kind':target['kind'],
           'runtime_dispatch_confirmed':False,
           **initializer_site_context(phase,owner),
           **literal_call_arity_evidence(argument_count,target,_source_platform(path)),
           'evidence':graphs[path]['_evidence'](part.line,part.col,end_line=part.end_line,end_column=part.end_col)})
    for path,owner,local,refer,part,phase,argument_count in pending_referred_calls:
        if refer is None:
            unresolved(path,'unqualified_referred_call',local,part.line,part.col,
                       'ambiguous_or_conflicting_refer_local_name')
            continue
        target_ns,member=refer
        # A definition with the same unqualified name in the current ns is
        # not evidence of an external call. It may be a conflict/rebinding.
        source_ns=files[path]['namespace'][0]
        if source_ns and def_index.get((source_ns[0],local)):
            unresolved(path,'unqualified_referred_call',local,part.line,part.col,
                       'same_namespace_declaration_may_shadow_refer')
            continue
        ns_target,reason=namespace_target(path,target_ns)
        if ns_target is None:
            unresolved(path,'unqualified_referred_call',local,part.line,part.col,reason)
            continue
        targets=[n for n in def_index.get((target_ns,member),[])
                 if n['path']==ns_target['path'] and n['kind'] in {'function','macro','multimethod','protocol_method_declaration'}]
        if len(targets)!=1:
            unresolved(path,'unqualified_referred_call',local,part.line,part.col,
                       'target_not_callable_or_not_indexed' if not targets else 'ambiguous_callable_definitions')
            continue
        target=targets[0]
        if target['kind']=='protocol_method_declaration':
            unresolved(path,'unqualified_referred_call',local,
                part.line,part.col,'protocol_runtime_dispatch_unresolved')
        if target['path']!=path and target.get('visibility')!='public':
            unresolved(path,'unqualified_referred_call',local,part.line,part.col,
                       'private_var_not_public_cross_namespace' if target.get('visibility')=='private'
                       else 'target_var_visibility_not_statistically_verified')
            continue
        is_refer_all = files[path].get('refer_all_materialized_bindings', {}).get(local) == (target_ns, member)
        if phase=='macro_expansion':relation='clojure_referred_macro_phase_candidate'
        elif target['kind']=='macro':relation='clojure_referred_macro_invocation_candidate'
        elif target['kind']=='protocol_method_declaration':relation='clojure_referred_protocol_call_candidate'
        else:relation='clojure_referred_call_candidate'
        graphs[path]['edges'].append({
            'source':owner['id'],'target':target['id'],
            'source_path':path,'target_path':target['path'],
            'relation':relation,'resolution':('clojure_source_project_refer_all_candidate' if is_refer_all
                                              else 'clojure_literal_refer_scope_candidate'),
            'refer_binding_provenance':('unique_literal_public_project_source_not_compiler_ns_publics'
                                        if is_refer_all else 'literal_explicit_refer_libspec'),
            'analyzer':'clojure-form-reader', 'source_platform':_source_platform(path),
            'target_platform':_source_platform(target['path']),
            'analysis_phase':phase, 'target_declaration_kind':target['kind'],
            'runtime_dispatch_confirmed':False,
            'refer_local_name':local, 'refer_original_name':member,
            'lexical_scope_analysis':'bounded_subset_lexical_guard_not_certified',
            **initializer_site_context(phase,owner),
            **literal_call_arity_evidence(argument_count,target,_source_platform(path)),
            'evidence':graphs[path]['_evidence'](part.line,part.col,end_line=part.end_line,end_column=part.end_col),
        })
    # Value-use resolution shares the original namespace/classpath selection
    # and private Var gates; unknown or conflicting bindings are not edges.
    seen_value_refs=set()
    for path,owner,part,phase,reference_form in pending_value_references:
        symbol=part.value
        namespace=files[path]['namespace'][0]
        if not namespace:
            continue
        current_ns,aliases=namespace[0],files[path]['namespace'][1]
        member=symbol
        if '/' in symbol:
            prefix,member=symbol.split('/',1)
            if prefix in files[path]['alias_guards']:
                unresolved(path,'qualified_value_reference',symbol,part.line,part.col,
                           files[path]['alias_guards'][prefix]);continue
            if (prefix not in aliases and prefix!=current_ns and '.' not in prefix
                    and prefix not in namespace_names):
                continue
            target_ns=aliases.get(prefix,prefix)
            original_kind='qualified_value_reference'
        elif symbol in files[path]['referred_names'][0] and not def_index.get((current_ns,symbol)):
            foreign=files[path]['referred_names'][0][symbol]
            if foreign is None:
                unresolved(path,'referred_value_reference',symbol,part.line,part.col,
                           'ambiguous_or_conflicting_refer_local_name');continue
            target_ns,member=foreign
            original_kind='referred_value_reference'
        else:
            target_ns=current_ns
            original_kind='local_namespace_value_reference'
        ns_target,reason=namespace_target(path,target_ns)
        if ns_target is None:
            unresolved(path,original_kind,symbol,part.line,part.col,reason)
            continue
        targets=[n for n in def_index.get((target_ns,member),[])
                 if n['path']==ns_target['path'] and n['kind'] in
                 {'variable','function','macro','multimethod','protocol','record','type','interface','protocol_method_declaration'}]
        if len(targets)!=1:
            unresolved(path,original_kind,symbol,part.line,part.col,
                'target_declaration_not_indexed' if not targets else 'ambiguous_var_declarations')
            continue
        target=targets[0]
        # A macro Var is accessible through (var m) or #'m; taking its
        # *ordinary evaluated symbol value* is rejected by the Clojure/JVM
        # compiler.  Do not publish an apparently usable data reference.
        if (target['kind']=='macro' and reference_form=='ordinary_symbol'
                and _source_platform(path)=='clj'):
            unresolved(path,original_kind,symbol,part.line,part.col,
                       'macro_symbol_cannot_be_used_as_ordinary_runtime_value')
            continue
        if target['path']!=path and target.get('visibility')!='public':
            unresolved(path,original_kind,symbol,part.line,part.col,
                'private_var_not_public_cross_namespace' if target.get('visibility')=='private'
                else 'target_var_visibility_not_statistically_verified')
            continue
        signature=(owner['id'],target['id'],part.line,part.col,phase)
        if signature in seen_value_refs:continue
        seen_value_refs.add(signature)
        graphs[path]['edges'].append({
            'source':owner['id'],'target':target['id'],
            'source_path':path,'target_path':target['path'],
            'relation':'clojure_var_value_reference_candidate',
            'resolution':'literal_var_declaration_identity_not_compiler_binding',
            'analyzer':'clojure-static-scoped-value-index',
            'source_platform':_source_platform(path),
            'target_platform':_source_platform(target['path']),
            'analysis_phase':phase,'target_declaration_kind':target['kind'],
            **initializer_site_context(phase,owner),
            'reference_syntax':original_kind,
            'reference_binding_provenance':('source_project_refer_all_candidate_only'
                if (original_kind == 'referred_value_reference' and
                    files[path].get('refer_all_materialized_bindings', {}).get(symbol) == (target_ns, member))
                else 'explicit_refer_or_local_source_candidate'),
            'reference_form':reference_form,
            'reference_semantics':('var_object_not_value_dereference'
                if reference_form in {'var_special_form','var_quote'}
                else 'ordinary_symbol_value_candidate'),
            # CLJS compiler emits a compile-time-metadata Var reflector,
            # unlike JVM Clojure which uses a runtime reified Var.  Shared
            # (.cljc) syntax must not pick either platform by assumption.
            'var_platform_semantics':(
                ('cljs_compile_time_metadata_reflector_candidate'
                 if _source_platform(path)=='cljs' else
                 'jvm_var_object_candidate'
                 if _source_platform(path)=='clj' else
                 'shared_platform_var_semantics_not_selected')
                if reference_form in {'var_special_form','var_quote'}
                else 'ordinary_symbol_value_candidate'),
            'compiler_binding_confirmed':False,
            'call_binding_confirmed':False,
            'evidence':graphs[path]['_evidence'](part.line,part.col,
                end_line=part.end_line,end_column=part.end_col)})

    # Bytecode declarations from an explicitly selected, checksum-validated
    # effective classpath. No class loading, analyzer macro execution, or
    # runtime dispatch is attempted. This tier is deliberately JVM-only;
    # .cljc/CLJS contexts must not inherit JVM interop as compiler proof.
    from .clojure_jvm_bytecode import analyze_jvm_interop
    from .clojure_core_context import shadowed_core_names, core_macro_aliases
    for path, data in files.items():
        platform = _source_platform(path)
        if platform not in ('clj', 'shared'):
            continue
        # Shared sources require an *independent JVM projection*. Never
        # inspect the unprojected .cljc syntax as an unqualified JVM program.
        if platform == 'shared':
            forms = data['_platform_projected_forms'].get('clj')
            if forms is None:
                graphs[path]['cljc_jvm_interop_evidence'] = {
                    'status': 'unavailable_platform_projection',
                    'source_platform_view': 'clj',
                    'reason': data['platform_projections'].get('clj',{}).get('reason',
                               'jvm_platform_projection_not_verified')}
                continue
            namespace = _namespace(list(forms), 'clj')
            alias_guards = _namespace_alias_guards(list(forms), 'clj')
            referred = _explicit_refers(list(forms), 'clj')[0]
        else:
            forms = data['forms']
            namespace = data['namespace']
            alias_guards = data['alias_guards']
            referred = data['referred_names'][0]
        snapshot = source_contexts[path].get('externally_supplied_classpath')
        analysis = analyze_jvm_interop(
            forms, snapshot, aliases=namespace[1], alias_guards=alias_guards,
            referred=referred,shadowed_core=shadowed_core_names(forms,referred),
            macro_aliases=core_macro_aliases(forms),
            target_jvm_major=_graph.get('_atlas_clojure_target_jvm') if isinstance(_graph,dict) else None)
        if platform == 'clj':
            graphs[path]['jvm_bytecode_interop_evidence'] = analysis
        else:
            graphs[path]['cljc_jvm_interop_evidence'] = {
                'status':'platform_scoped_candidate_not_compiler_verified',
                'source_platform_view':'clj',
                'read_mode':':read-cond :allow',
                'source_projection':data['platform_projections']['clj']['status'],
                'analysis':analysis}

    # ClojureScript JS evidence comes only from ClojureScript expressions,
    # including the explicitly projected :cljs branch of a .cljc file.
    # These two platform products never assert the same runtime context.
    from .clojurescript_js_interop import analyze_cljs_js_interop
    from .clojurescript_js_modules import analyze_cljs_module_imports
    for path,data in files.items():
        platform = _source_platform(path)
        if platform not in ('cljs','shared'):
            continue
        if platform == 'shared':
            forms = data['_platform_projected_forms'].get('cljs')
            if forms is None:
                graphs[path]['cljc_js_module_evidence'] = {
                    'status':'unavailable_platform_projection',
                    'source_platform_view':'cljs',
                    'reason':data['platform_projections'].get('cljs',{}).get('reason',
                              'cljs_platform_projection_not_verified')}
                graphs[path]['cljc_js_interop_evidence'] = {
                    'status':'unavailable_platform_projection',
                    'source_platform_view':'cljs',
                    'reason':data['platform_projections'].get('cljs',{}).get('reason',
                              'cljs_platform_projection_not_verified')}
                continue
            namespace = _namespace(list(forms),'cljs')
            alias_guards = _namespace_alias_guards(list(forms),'cljs')
            referred = _explicit_refers(list(forms),'cljs')[0]
        else:
            forms = data['forms']
            namespace = data['namespace']
            alias_guards = data['alias_guards']
            referred = data['referred_names'][0]
        analysis = analyze_cljs_js_interop(
            forms,aliases=namespace[1],alias_guards=alias_guards,
            shadowed_core=shadowed_core_names(forms,referred),
            macro_aliases=core_macro_aliases(forms))
        module_evidence = analyze_cljs_module_imports(
            forms,source_path=path,root=root)
        if platform == 'cljs':
            graphs[path]['cljs_js_module_evidence'] = module_evidence
            graphs[path]['cljs_js_interop_evidence'] = analysis
        else:
            graphs[path]['cljc_js_module_evidence'] = {
                'status':'platform_scoped_module_inventory_not_compiler_verified',
                'source_platform_view':'cljs',
                'read_mode':':read-cond :allow',
                'source_projection':data['platform_projections']['cljs']['status'],
                'analysis':module_evidence}
            graphs[path]['cljc_js_interop_evidence'] = {
                'status':'platform_scoped_candidate_not_compiler_verified',
                'source_platform_view':'cljs',
                'read_mode':':read-cond :allow',
                'source_projection':data['platform_projections']['cljs']['status'],
                'analysis':analysis}

    # Source-anchored caller argument -> callee parameter candidates. This
    # enriches existing guarded call edges without inventing a runtime target.
    from .clojure_type_resolution import connect_project_type_hints
    connect_project_type_hints(graphs, files)
    from .clojure_interprocedural import attach_interprocedural_argument_candidates
    attach_interprocedural_argument_candidates(graphs, files)

    # Optional EXTERNAL compiler report is reconciled only against independently
    # derived candidate edges. It never runs compiler or project code and does
    # not promote an untrusted report into a verified binding.
    from .clojure_compiler_report import reconcile_compiler_report
    reconcile_compiler_report(graphs, files, _graph if isinstance(_graph, dict) else {}, root)

    for source_path,g in graphs.items():
        for edge in g['edges']:
            target_path=edge.get('target_path')
            if not target_path or not edge.get('relation','').startswith('clojure_'):
                continue
            declaration=files.get(target_path,{}).get('namespace',(None,))[0]
            if declaration:
                proof=source_path_evidence(source_contexts[source_path],declaration[0],
                    _source_platform(source_path),target_path)
                if proof:edge['project_source_path_evidence']=proof
        g.pop('_evidence',None)
    return graphs,warnings
