"""Offline SQL structural subset parser and evidence-first table graph.

No SQL engine, driver, server, build system, code execution or network access.
This is deliberately NOT a complete SQL grammar or a name-binding resolver.
Only bounded CREATE TABLE/VIEW, SELECT, INSERT, UPDATE, DELETE, ALTER TABLE
and certain FOREIGN KEY constructs are structurally recognized. SQL dialects,
search paths, migrations and runtime database state remain unresolved.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
from typing import NamedTuple

from analyzer.symbol_graph import _file_id, _node_id

MAX_FILES = 400
MAX_FILE_BYTES = 1_000_000
MAX_TOKENS = 45000
MAX_STATEMENTS = 2000


class SQLLexError(ValueError):
    pass


class Token(NamedTuple):
    kind: str
    value: str
    line: int
    column: int
    start: int
    end: int

    @property
    def keyword(self):
        return self.value.upper() if self.kind == 'word' else ''


@dataclass(frozen=True)
class Identifier:
    values: tuple[str, ...]
    quotes: tuple[bool, ...]
    line: int
    column: int
    end_index: int

    @property
    def name(self):
        return '.'.join(self.values)

    @property
    def key(self):
        # Avoid equating a case-sensitive quoted identifier with an unquoted
        # token: dialect-specific case folding cannot be established here.
        return tuple((v if q else v.casefold(), q) for v, q in zip(self.values, self.quotes))


# A word in these positions is syntax, not a SQL table identifier.
_NOT_TABLE = {
    'SELECT', 'WHERE', 'FROM', 'JOIN', 'INNER', 'OUTER', 'LEFT', 'RIGHT',
    'FULL', 'CROSS', 'NATURAL', 'ON', 'USING', 'AS', 'WITH', 'RECURSIVE',
    'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'VALUES', 'SET', 'RETURNING',
    'TABLE', 'UNNEST', 'LATERAL', 'JSON_TABLE', 'CASE', 'WHEN', 'END',
    'NULL', 'TRUE', 'FALSE', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
    'OVER', 'PARTITION', 'WINDOW', 'QUALIFY', 'UNION', 'EXCEPT', 'INTERSECT',
}
_CLAUSE_END = {
    'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'RETURNING',
    'ON', 'USING', 'WINDOW', 'QUALIFY', 'UNION', 'EXCEPT', 'INTERSECT',
    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL', 'CROSS', 'NATURAL',
    'SET', 'VALUES', 'FOR', 'CONNECT', 'START',
}


def tokenize_sql(text: str) -> list[Token]:
    """SQL-aware lexical tokens with original line and column; never execute SQL.

    Skips SQL comments, string literals, and PostgreSQL dollar-quoted bodies.
    Unknown dialect tokens may be skipped by the structural parser, never
    promoted to a confirmed reference.
    """
    if '\x00' in text:
        raise SQLLexError('SQL source contains NUL')
    tokens: list[Token] = []
    n = len(text)
    i, line, col = 0, 1, 1

    def advance(start, stop):
        nonlocal line, col
        chunk = text[start:stop]
        breaks = chunk.count('\n')
        if breaks:
            line += breaks
            col = len(chunk.rsplit('\n', 1)[-1]) + 1
        else:
            col += len(chunk)

    def emit(kind, value, start, stop, ln, column):
        tokens.append(Token(kind, value, ln, column, start, stop))
        if len(tokens) > MAX_TOKENS:
            raise SQLLexError('SQL token limit exceeded')

    while i < n:
        c = text[i]
        if c.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            advance(i, j); i = j
            continue
        if text.startswith('--', i) or c == '#':
            j = text.find('\n', i)
            if j == -1:
                j = n
            advance(i, j); i = j
            continue
        if text.startswith('/*', i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith('/*', j):
                    depth += 1; j += 2
                elif text.startswith('*/', j):
                    depth -= 1; j += 2
                else:
                    j += 1
            if depth:
                raise SQLLexError(f'unterminated SQL comment at {line}:{col}')
            advance(i, j); i = j
            continue
        start, ln, column = i, line, col
        if c == '$':
            match = re.compile(r'\$[A-Za-z_0-9]*\$').match(text, i)
            if match:
                delimiter = match.group()
                j = text.find(delimiter, i + len(delimiter))
                if j < 0:
                    raise SQLLexError(f'unterminated dollar quote at {line}:{col}')
                j += len(delimiter)
                emit('string', '', start, j, ln, column)
                advance(i, j); i = j
                continue
        if c in ("'", '"', '`', '['):
            closing = ']' if c == '[' else c
            j = i + 1
            chars = []
            finished = False
            while j < n:
                if text[j] == closing:
                    if j + 1 < n and text[j + 1] == closing:
                        chars.append(closing); j += 2; continue
                    j += 1; finished = True; break
                if c == "'" and text[j] == '\\' and j + 1 < n:
                    # A backslash-escaped quote in MySQL literals must not be
                    # mistaken for the end of a string.
                    chars.append(text[j + 1]); j += 2; continue
                chars.append(text[j]); j += 1
            if not finished:
                raise SQLLexError(f'unterminated SQL literal/identifier at {line}:{col}')
            emit('string' if c == "'" else 'identifier', ''.join(chars), start, j, ln, column)
            advance(i, j); i = j
            continue
        if c.isalpha() or c == '_':
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in ('_', '$')):
                j += 1
            emit('word', text[i:j], start, j, ln, column)
            advance(i, j); i = j
            continue
        if c.isdigit():
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] == '.'):
                j += 1
            emit('number', text[i:j], start, j, ln, column)
            advance(i, j); i = j
            continue
        emit('punct', c, start, i + 1, ln, column)
        advance(i, i + 1); i += 1
    return tokens


def split_statements(tokens: list[Token]) -> list[list[Token]]:
    groups = []
    start, depth = 0, 0
    for i, token in enumerate(tokens):
        if token.value == '(' and token.kind == 'punct':
            depth += 1
        elif token.value == ')' and token.kind == 'punct':
            depth -= 1
            if depth < 0:
                raise SQLLexError(f'unbalanced right parenthesis at {token.line}:{token.column}')
        elif token.value == ';' and depth == 0:
            if i > start:
                groups.append(tokens[start:i])
            start = i + 1
    if depth:
        raise SQLLexError('unclosed SQL parenthesis')
    if start < len(tokens):
        groups.append(tokens[start:])
    if len(groups) > MAX_STATEMENTS:
        raise SQLLexError('SQL statement limit exceeded')
    return groups


def _identifier(tokens: list[Token], i: int) -> Identifier | None:
    if i >= len(tokens):
        return None
    values, quotes = [], []
    start = tokens[i]
    while i < len(tokens):
        t = tokens[i]
        if t.kind not in ('word', 'identifier') or (t.kind == 'word' and t.keyword in _NOT_TABLE):
            break
        values.append(t.value)
        quotes.append(t.kind == 'identifier')
        i += 1
        if i >= len(tokens) or tokens[i].value != '.':
            break
        if i + 1 >= len(tokens) or tokens[i + 1].kind not in ('word', 'identifier'):
            return None
        i += 1
    if not values:
        return None
    return Identifier(tuple(values), tuple(quotes), start.line, start.column, i)


def _cte_names(tokens: list[Token]) -> tuple[set[tuple], bool]:
    """Extract WITH aliases, otherwise reject potentially ambiguous CTEs.

    Supports WITH [RECURSIVE] name [column-list] AS (query), ...
    """
    ctes: set[tuple] = set()
    for pos, token in enumerate(tokens):
        if token.keyword != 'WITH':
            continue
        i = pos + 1
        if i < len(tokens) and tokens[i].keyword == 'RECURSIVE':
            i += 1
        # WITH in comments/strings is not tokenized as a keyword.
        if i >= len(tokens) or tokens[i].kind not in ('word','identifier'):
            return set(), False
        while i < len(tokens):
            cte = _identifier(tokens, i)
            if not cte or len(cte.values) != 1:
                return set(), False
            ctes.add(cte.key)
            i = cte.end_index
            if i < len(tokens) and tokens[i].value == '(':
                depth = 1; i += 1
                while i < len(tokens) and depth:
                    if tokens[i].value == '(': depth += 1
                    if tokens[i].value == ')': depth -= 1
                    i += 1
            if i >= len(tokens) or tokens[i].keyword != 'AS':
                return set(), False
            i += 1
            if i < len(tokens) and tokens[i].keyword == 'NOT':
                i += 1
                if i < len(tokens) and tokens[i].keyword == 'MATERIALIZED': i += 1
            elif i < len(tokens) and tokens[i].keyword == 'MATERIALIZED':
                i += 1
            if i >= len(tokens) or tokens[i].value != '(':
                return set(), False
            depth = 1; i += 1
            while i < len(tokens) and depth:
                if tokens[i].value == '(': depth += 1
                elif tokens[i].value == ')': depth -= 1
                i += 1
            if depth:
                return set(), False
            if i < len(tokens) and tokens[i].value == ',':
                i += 1
                continue
            break
    return ctes, True


def _head(tokens: list[Token]) -> tuple[str, Identifier | None]:
    """Recognize a bounded structural SQL statement, not a full grammar."""
    if not tokens:
        return 'unknown', None
    i = 0
    if tokens[0].keyword == 'WITH':
        # Outer statement is a CTE query; scanning all FROM/JOIN references is
        # safe once the WITH structure was checked separately.
        return 'with_query', None
    first = tokens[0].keyword
    if first == 'CREATE':
        i = 1
        if i < len(tokens) and tokens[i].keyword == 'OR':
            i += 1
            if i < len(tokens) and tokens[i].keyword == 'REPLACE': i += 1
        while i < len(tokens) and tokens[i].keyword in ('TEMP', 'TEMPORARY', 'UNLOGGED'):
            i += 1
        if i >= len(tokens) or tokens[i].keyword not in ('TABLE','VIEW'):
            return 'unknown', None
        kind = 'create_' + tokens[i].keyword.lower()
        i += 1
        if i + 2 < len(tokens) and [x.keyword for x in tokens[i:i+3]] == ['IF','NOT','EXISTS']:
            i += 3
        return kind, _identifier(tokens, i)
    if first == 'ALTER' and len(tokens)>2 and tokens[1].keyword == 'TABLE':
        return 'alter_table', _identifier(tokens, 2)
    if first == 'INSERT':
        if len(tokens)>2 and tokens[1].keyword == 'INTO':
            return 'insert', _identifier(tokens,2)
        return 'unknown', None
    if first == 'UPDATE':
        return 'update', _identifier(tokens,1)
    if first == 'DELETE' and len(tokens)>2 and tokens[1].keyword=='FROM':
        return 'delete', _identifier(tokens,2)
    if first == 'SELECT':
        return 'select', None
    return 'unknown', None


def _references(tokens: list[Token], kind: str, declared: Identifier | None, ctes: set[tuple]):
    """Return (SQL identifier, role) for syntax-identifiable table sites."""
    refs = []
    if kind in ('insert','update','delete','alter_table') and declared:
        refs.append((declared, 'writes_table'))
    for i, token in enumerate(tokens):
        keyword = token.keyword
        if keyword not in ('FROM','JOIN','REFERENCES'):
            continue
        if kind == 'delete' and keyword == 'FROM' and i == 1:
            continue
        if keyword == 'FROM' and i and tokens[i-1].keyword == 'DELETE':
            continue
        start = i + 1
        if start < len(tokens) and tokens[start].keyword in ('ONLY','LATERAL'):
            start += 1
        ident = _identifier(tokens, start)
        if ident and (keyword == 'REFERENCES' or not (ident.end_index < len(tokens) and tokens[ident.end_index].value == '(')):
            if ident.key not in ctes:
                refs.append((ident, 'foreign_key' if keyword == 'REFERENCES' else 'reads_table'))
        # Recognize simple comma joins with or without a literal alias:
        # FROM accounts AS a, ledger l. The parser still deliberately rejects
        # expression-based/generated table names and table-valued functions.
        if keyword == 'FROM' and ident:
            j = ident.end_index
            while j < len(tokens):
                if tokens[j].keyword == 'AS':
                    alias = _identifier(tokens, j + 1)
                    if not alias or len(alias.values) != 1:
                        break
                    j = alias.end_index
                elif tokens[j].kind in ('word', 'identifier') and tokens[j].keyword not in _CLAUSE_END:
                    alias = _identifier(tokens, j)
                    if alias and len(alias.values) == 1:
                        j = alias.end_index
                if j >= len(tokens) or tokens[j].value != ',':
                    break
                following = _identifier(tokens, j + 1)
                if not following or (following.end_index < len(tokens) and tokens[following.end_index].value == '('):
                    break
                if following.key not in ctes:
                    refs.append((following, 'reads_table'))
                j = following.end_index
    # Unsupported FROM expressions remain unlinked rather than guessed.
    seen = set(); out = []
    for ident, role in refs:
        sig = (ident.line, ident.column, ident.key, role)
        if sig not in seen:
            seen.add(sig); out.append((ident, role))
    return out


def _load(root: Path, candidates: dict) -> tuple[dict, list]:
    root = root.resolve()
    files = {}
    warnings = []
    names = sorted(p for p, info in candidates.items() if info.get('language') == 'sql')
    for path in names[:MAX_FILES]:
        f = root / path
        try:
            if f.is_symlink() or not f.resolve().is_relative_to(root) or not f.is_file():
                raise ValueError('path_missing_symlink_or_outside_root')
            if f.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('source_exceeds_sql_limit')
            data = f.read_bytes()
            text = data.decode('utf-8-sig')
            tokens = tokenize_sql(text)
            groups = split_statements(tokens)
            files[path] = {'sha': hashlib.sha256(data).hexdigest(),
                           'lines':text.splitlines(), 'statements':groups}
        except (OSError, UnicodeError, SQLLexError, ValueError) as error:
            warnings.append({'path':path,'reason':'sql_source_unavailable_or_invalid_subset',
                             'details':str(error)[:180]})
    if len(names) > MAX_FILES:
        warnings.append({'reason':'sql_file_limit', 'found':len(names),'limit':MAX_FILES})
    return files, warnings


def analyze_sql(root: Path, candidates: dict, _graph: dict) -> tuple[dict, list]:
    """Create source-line evidenced declarations and conservative SQL candidates."""
    files, warnings = _load(root, candidates)
    graphs = {}
    definitions = {}
    pending = []
    for path, item in sorted(files.items()):
        sha, lines = item['sha'], item['lines']
        file_node = {'id':_file_id(path),'path':path,'name':Path(path).name,
                     'kind':'file','line':1,'confidence':'verified_file'}
        graph = {'nodes':[file_node], 'edges':[], 'hash':sha,
                 'parser':'sql_structural_subset',
                 'caution':'Offline SQL lexer + selected structural syntax only. '
                           'Not a validated full SQL grammar, dialect checker, database schema resolver '
                           'or proof of migrations/runtime queries.'}
        graphs[path] = graph

        def evidence(line, col=0):
            return {'path':path,'line':line,'column':col,'source_sha256':sha,
                    'snippet':lines[line-1].strip()[:200] if 0 < line <= len(lines) else ''}

        for statement in item['statements']:
            kind, declaration = _head(statement)
            if kind == 'unknown' or (kind in ('create_table','create_view','insert','update','delete','alter_table') and declaration is None):
                continue
            ctes, cte_valid = _cte_names(statement)
            if not cte_valid:
                warnings.append({'path':path,'line':statement[0].line,'reason':'sql_unsupported_cte'})
                continue
            first = statement[0]
            if declaration and kind.startswith('create_'):
                node_kind = 'table' if kind == 'create_table' else 'view'
                symbol = declaration
            else:
                node_kind = 'sql_statement'
                symbol = None
            qual = (symbol.name if symbol else f'{kind}@{first.start}')
            line = symbol.line if symbol else first.line
            column = symbol.column if symbol else first.column
            sid = _node_id(path, qual, line, column)
            owner = {'id':sid,'path':path,'name':qual if symbol else kind.upper(),
                     'qualname':qual,'kind':node_kind,'line':line,'column':column,
                     'parent':file_node['id'],'confidence':'parsed_structural_subset'}
            graph['nodes'].append(owner)
            graph['edges'].append({'source':file_node['id'],'target':sid,
                'source_path':path,'target_path':path,'relation':'declares',
                'resolution':'parsed_structural_subset','analyzer':'sql-structural-subset',
                'evidence':evidence(line,column)})
            if symbol:
                definitions.setdefault(symbol.key,[]).append(owner)
            for ref, role in _references(statement, kind, declaration, ctes):
                pending.append((path, owner, ref, role))

    for path, owner, ref, role in pending:
        targets = definitions.get(ref.key, [])
        if len(targets) != 1:
            continue  # Duplicate/different schema definitions are ambiguous.
        target = targets[0]
        if owner['id'] == target['id']:
            continue
        lines, sha = files[path]['lines'], files[path]['sha']
        graphs[path]['edges'].append({
            'source':owner['id'],'target':target['id'],
            'source_path':path,'target_path':target['path'],
            'relation':'sql_table_candidate',
            'resolution':'sql_unique_decl_candidate',
            'analyzer':'sql-structural-subset',
            'sql_role':role,
            'evidence':{'path':path,'line':ref.line,'column':ref.column,
                        'source_sha256':sha,
                        'snippet':lines[ref.line-1].strip()[:200] if 0<ref.line<=len(lines) else ''},
        })
    return graphs, warnings
