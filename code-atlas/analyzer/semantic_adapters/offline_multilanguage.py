"""Read-only, source-anchored lexical/structural evidence for underserved languages.

This is NOT a full grammar/type checker and NEVER resolves calls from name equality.
Its language-specific rules recognize *only* unambiguous declaration and literal
module-reference shapes; dynamic constructs and unsupported syntax stay unresolved.
No project code, parser hook, plugin, package manager, compiler or subprocess runs.

A shared bounded character scanner makes source spans, comments, strings, and
UTF-8 hashes consistent; distinct language matchers interpret the actual tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import posixpath
import re

from analyzer.symbol_graph import _file_id, _node_id

MAX_BYTES = 1_500_000
MAX_TOKENS = 150_000
MAX_REFERENCES = 10000


@dataclass(frozen=True)
class Token:
    text: str
    kind: str
    line: int
    col: int
    end_line: int
    end_col: int
    offset: int


class ScanError(ValueError):
    pass


_LUA_LONG_COMMENT = re.compile(r'--\[(=*)\[')
_LUA_LONG_STRING = re.compile(r'\[(=*)\[')
_RUST_RAW_STRING = re.compile(r'r(\#*)"')


# Syntax-specific comment / string policies. These are lexer rules, not guessed
# source text regular expressions. This table intentionally excludes languages
# that already have stronger, independently maintained semantic adapters.
PROFILES = {
    'shell': dict(line=('#',), block=(), quote=('"',"'")),
    'rust': dict(line=('//',), block=(('/*','*/',True),), quote=('"',"'")),
    'csharp': dict(line=('//',), block=(('/*','*/',False),), quote=('"',"'")),
    'dart': dict(line=('//',), block=(('/*','*/',True),), quote=('"',"'")),
    'scala': dict(line=('//',), block=(('/*','*/',True),), quote=('"',"'")),
    'groovy': dict(line=('//',), block=(('/*','*/',False),), quote=('"',"'")),
    'lua': dict(line=('--',), block=(), quote=('"',"'")),
    'r': dict(line=('#',), block=(), quote=('"',"'",'`')),
    'julia': dict(line=('#',), block=(('#=','=#',True),), quote=('"',"'",'`')),
    'perl': dict(line=('#',), block=(), quote=('"',"'")),
    'haskell': dict(line=('--',), block=(('{-','-}',True),), quote=('"',"'")),
    'erlang': dict(line=('%',), block=(), quote=('"',"'")),
    'elixir': dict(line=('#',), block=(), quote=('"',"'")),
    'fsharp': dict(line=('//',), block=(('(*','*)',True),), quote=('"',"'")),
    'visual_basic': dict(line=("'",), block=(), quote=('"',)),
    'zig': dict(line=('//',), block=(), quote=('"',"'")),
    'nim': dict(line=('#',), block=(('#[',']#',True),), quote=('"',"'")),
}


def tokenize(source: str, language: str) -> list[Token]:
    policy=PROFILES[language]
    tokens: list[Token]=[]
    n=len(source); i=0; line=1; col=1
    def consume(stop: int):
        nonlocal i,line,col
        fragment=source[i:stop]
        k=fragment.count('\n')
        if k:
            line+=k
            col=len(fragment.rsplit('\n',1)[-1])+1
        else:
            col+=len(fragment)
        i=stop
    def emit(value: str, kind: str, stop: int):
        ln,cl,start=line,col,i
        consume(stop)
        if len(tokens)>=MAX_TOKENS:raise ScanError('token_limit')
        tokens.append(Token(value,kind,ln,cl,line,col,start))
    while i<n:
        ch=source[i]
        if ch.isspace():consume(i+1);continue
        # Perl POD/data sections and Zig multiline string lines are source
        # content, never declaration text. Handle before ordinary tokens.
        if language=='visual_basic' and source[i:i+3].lower()=='rem' and (i+3==n or source[i+3].isspace()):
            prefix=source[source.rfind('\n',0,i)+1:i]
            if not prefix.strip() or prefix.rstrip().endswith(':'):
                eol=source.find('\n',i)
                consume(n if eol<0 else eol);continue
        if language=='perl' and col==1 and (source.startswith('__DATA__',i) or source.startswith('__END__',i)):
            consume(n);break
        if language=='perl' and col==1 and any(source.startswith(tag,i) for tag in ('=pod','=head','=begin','=over','=item','=encoding','=for')):
            end=re.search(r'(?m)^=cut\b[^\n]*',source[i:])
            if not end:raise ScanError('unterminated_perl_pod')
            consume(i+end.end());continue
        if language=='zig' and source.startswith('\\\\',i) and not source[source.rfind('\n',0,i)+1:i].strip():
            end=source.find('\n',i)
            consume(n if end<0 else end);continue
        if language=='shell' and source.startswith('<<',i):
            # Here-documents contain arbitrary text. Treating their body as
            # executable shell statements would create invented declarations.
            m=re.match(r'<<(-?)([\'"]?)([A-Za-z_][A-Za-z_0-9]*)\2',source[i:])
            if not m:raise ScanError('unsupported_shell_heredoc')
            delim=m.group(3);allow_tabs=bool(m.group(1))
            eol=source.find('\n',i+len(m.group(0)))
            if eol<0:raise ScanError('unterminated_shell_heredoc')
            cursor=eol+1
            while cursor<n:
                end=source.find('\n',cursor)
                if end<0:end=n
                content=source[cursor:end].lstrip('\t') if allow_tabs else source[cursor:end]
                if content.strip('\r')==delim:
                    consume(end);break
                cursor=end+1
            else:
                raise ScanError('unterminated_shell_heredoc')
            continue
        if language=='rust' and source.startswith('r',i):
            raw=_RUST_RAW_STRING.match(source,i)
            if raw:
                close='"'+raw.group(1)
                j=source.find(close,i+len(raw.group(0)))
                if j<0:raise ScanError('unterminated_rust_raw_string')
                emit(source[i+len(raw.group(0)):j],'string',j+len(close));continue
        if language=='rust' and ch=="'" and i+1<n and source[i+1].isalpha():
            # 'a and 'static are lifetime identifiers, not unterminated chars.
            # Actual character literals have a closing quote.
            match=re.match(r"'(?:\\.|[^'\\\n])'",source[i:])
            if not match:emit("'",'punct',i+1);continue
        # Lua long-bracket strings / comments: exact matching '=' level.
        if language=='lua':
            opener=_LUA_LONG_COMMENT.match(source,i) if source.startswith('--[',i) else None
            if opener:
                close=']'+opener.group(1)+']'
                finish=source.find(close,i+len(opener.group(0)))
                if finish<0:raise ScanError('unterminated_lua_long_comment')
                consume(finish+len(close));continue
            opener=_LUA_LONG_STRING.match(source,i) if ch=='[' else None
            if opener:
                close=']'+opener.group(1)+']'
                finish=source.find(close,i+len(opener.group(0)))
                if finish<0:raise ScanError('unterminated_lua_long_string')
                emit(source[i+len(opener.group(0)):finish],'string',finish+len(close));continue
        # Check nested/multiline comments BEFORE single-line comments:
        # Julia #=...=# and Nim #[...]# start with the line-comment marker.
        matched=False
        for opening,closing,nested in policy['block']:
            if not source.startswith(opening,i):continue
            depth=1;j=i+len(opening)
            while j<n and depth:
                if nested and source.startswith(opening,j):depth+=1;j+=len(opening)
                elif source.startswith(closing,j):depth-=1;j+=len(closing)
                else:j+=1
            if depth:raise ScanError('unterminated_block_comment')
            consume(j);matched=True;break
        if matched:continue
        for marker in policy['line']:
            if source.startswith(marker,i):
                end=source.find('\n',i)
                consume(n if end<0 else end)
                matched=True;break
        if matched:continue
        # Groovy slashy regex strings can contain language keywords. Only
        # recognize an unambiguously expression-prefixed slashy literal;
        # division is left untouched.
        if language=='groovy' and ch=='/' and tokens and tokens[-1].text in ('=','(', '[',',',':','return','~','=~','==~'):
            j=i+1
            while j<n and source[j]!='\n':
                if source[j]=='\\':j+=2;continue
                if source[j]=='/':break
                j+=1
            if j<n and source[j]=='/':
                emit(source[i+1:j],'string',j+1);continue
        # Ordinary and multiline strings, including C# verbatim strings and
        # triple-quoted Dart/Scala/Julia/Groovy/Nim literals. We deliberately
        # hide interpolation internals rather than fabricate call edges.
        verbatim = language=='csharp' and source.startswith('@"',i)
        quoted = ch in policy['quote'] or verbatim
        if quoted:
            start=i
            q='"' if verbatim else ch
            qstart=i+1 if verbatim else i
            triple=(q=='"' and source.startswith('"""',qstart)) or (q=="'" and language in ('dart','groovy') and source.startswith("'''",qstart))
            delim=q*3 if triple else q
            j=qstart+len(delim)
            while j<n:
                if source.startswith(delim,j):
                    if verbatim and not triple and source.startswith('""',j):j+=2;continue
                    break
                if source[j]=='\\' and not verbatim:j+=2;continue
                j+=1
            if j>=n:raise ScanError('unterminated_string')
            text=source[qstart+len(delim):j]
            emit(text,'string',j+len(delim));continue
        if ch.isalpha() or ch=='_' or ch=='$' or (ord(ch)>127 and ch.isidentifier()):
            j=i+1
            while j<n and (source[j].isalnum() or source[j] in '_$' or (ord(source[j])>127 and source[j].isidentifier())):j+=1
            emit(source[i:j],'word',j);continue
        if ch.isdigit():
            j=i+1
            while j<n and (source[j].isalnum() or source[j] in '_.'):j+=1
            emit(source[i:j],'number',j);continue
        special=next((s for s in ('::','=>','->','<-','|>','..','==','!=','<=','>=','?.','?:','#{','@"') if source.startswith(s,i)),None)
        if special:emit(special,'punct',i+len(special));continue
        emit(ch,'punct',i+1)
    return tokens


def _name(t: Token | None) -> bool:
    return t is not None and t.kind=='word' and not t.text[0].isdigit()


def _at(ts: list[Token], i: int, value: str) -> bool:
    return 0<=i<len(ts) and ts[i].text == value


def _literal_arg(ts: list[Token], i: int) -> tuple[str,int] | None:
    # invocation of a known static import primitive with literal string only
    if _at(ts,i,'(') and i+1<len(ts) and ts[i+1].kind=='string' and _at(ts,i+2,')'):
        return ts[i+1].text, i+1
    if i<len(ts) and ts[i].kind=='string':return ts[i].text,i
    return None


def _qualified(ts: list[Token], i: int, connectors: tuple[str,...]=('.',)) -> tuple[str,int]:
    if i>=len(ts) or not _name(ts[i]):return '',i
    out=[ts[i].text];j=i+1
    while j+1<len(ts) and ts[j].text in connectors and _name(ts[j+1]):
        out.append(ts[j].text+ts[j+1].text);j+=2
    return ''.join(out),j


def _rust_opaque_macro_tokens(ts: list[Token]) -> set[int]:
    """Hide macro token-tree bodies that are NOT Rust item declarations.

    A macro invocation is arbitrary tokens, not source-proven expanded Rust
    syntax. This prevents lexical "fn"/"struct" in a macro definition or
    invocation from being manufactured as actual declaration nodes. The macro
    definition name itself remains visible as a source declaration.
    """
    hidden: set[int] = set()
    pairs={'(':')','[':']','{':'}'}
    for i,t in enumerate(ts):
        if i in hidden or t.text!='!' or i==0 or not _name(ts[i-1]):
            continue
        opening=i+1
        if ts[i-1].text=='macro_rules' and opening<len(ts) and _name(ts[opening]):
            opening+=1
        if opening>=len(ts) or ts[opening].text not in pairs:
            continue
        stack=[];j=opening
        while j<len(ts):
            value=ts[j].text
            if value in pairs:stack.append(pairs[value])
            elif value in pairs.values():
                if not stack or value!=stack.pop():
                    raise ScanError('invalid_rust_macro_token_tree_delimiters')
                if not stack:break
            j+=1
        if stack:raise ScanError('unterminated_rust_macro_token_tree')
        hidden.update(range(opening,j+1))
    return hidden


def _rust_mod_path_attribute(ts: list[Token], mod_index: int) -> tuple[str | None, Token | None, bool]:
    """Find the exact top-level #[path = "..."] attribute adjacent to `mod`.

    Returns (literal path, lexical token, path-attribute-seen). Computed
    expressions are never treated as source-defined file paths.
    """
    j=mod_index-1
    while j>=0 and ts[j].text==']':
        depth=1;k=j-1
        while k>=0 and depth:
            if ts[k].text==']':depth+=1
            elif ts[k].text=='[':depth-=1
            k-=1
        if depth or k<0 or ts[k].text!='#':break
        content=ts[k+2:j]
        if content and content[0].text=='path':
            if len(content)==3 and content[1].text=='=' and content[2].kind=='string':
                return content[2].text,content[2],True
            return None,None,True
        j=k-1
    return None,None,False


def _declarations(ts: list[Token], language: str):
    """Emit grammar-constrained decl/import events, never semantic call claims.

    Only literal, standalone forms are promoted to module candidates. Events
    keep exact token location. Import names are *not* treated as function calls.
    """
    events=[]
    def add(kind: str, name: str, tok: Token, *, other: str='', import_style: str=''):
        if name:
            if len(events)>=MAX_REFERENCES:raise ScanError('reference_limit')
            events.append((kind,name,tok,other,import_style))
    def decl(kind,i):
        if i<len(ts) and _name(ts[i]):add(kind,ts[i].text,ts[i])
    brace_depth=0
    rust_opaque=_rust_opaque_macro_tokens(ts) if language=='rust' else set()
    for i,t in enumerate(ts):
        if i in rust_opaque: continue
        if t.text=='}':brace_depth=max(0,brace_depth-1)
        s=t.text;n=ts[i+1] if i+1<len(ts) else None
        p=ts[i-1] if i else None
        # Many declaration keywords can occur inside paths. Never parse a
        # member `foo.class` or `x::fn` as an independent declaration.
        preceded_path = p is not None and p.text in ('.','::',':')
        if language=='rust':
            if s in ('struct','enum','trait','union','type','fn','mod','macro_rules','const','static') and not preceded_path:
                j=i+1+(s=='macro_rules' and _at(ts,i+1,'!'))
                if _name(ts[j] if j<len(ts) else None):
                    if s=='mod' and not (_at(ts,j+1,';') or _at(ts,j+1,'{')):continue
                    decl('function' if s=='fn' else s,j)
                    if s=='mod' and _at(ts,j+1,';') and brace_depth==0:
                        explicit,attr_token,has_path=_rust_mod_path_attribute(ts,i)
                        if has_path:
                            if explicit is not None and attr_token is not None:
                                add('import',explicit,attr_token,import_style='rust_mod_path')
                        else:
                            add('import',ts[j].text,ts[j],import_style='rust_mod')
            if s=='use' and not preceded_path:
                # `use foo::bar;` is not necessarily a local file dependency.
                # Record textual reference only, never a guessed file target.
                q,j=_qualified(ts,i+1,('::',))
                if q:add('module_reference',q,t,import_style='rust_use')
        elif language=='csharp':
            if s in ('class','interface','struct','enum','record','delegate') and not preceded_path:
                j=i+1+(s=='record' and _at(ts,i+1,'struct'))
                decl(s,j)
            elif s=='namespace' and not preceded_path:
                q,j=_qualified(ts,i+1)
                if q:add('namespace',q,n or t)
            elif s=='using' and not preceded_path:
                j=i+1
                if _at(ts,j,'static'):j+=1
                if _name(ts[j] if j<len(ts) else None) and _at(ts,j+1,'='):j+=2
                q,j=_qualified(ts,j)
                if q and _at(ts,j,';'):add('module_reference',q,t,import_style='csharp_using')
        elif language=='dart':
            if s in ('class','mixin','enum','extension','typedef') and not preceded_path:
                decl(s,i+1)
            elif s in ('import','export','part') and n and n.kind=='string':
                # URI resolution is sensitive to pubspec, package: and parts.
                add('import',n.text,n,import_style='dart_uri')
            elif s=='library' and n and _name(n):
                q,_=_qualified(ts,i+1);add('namespace',q,n)
        elif language=='scala':
            if s in ('class','object','trait','enum','type','def','val','var') and not preceded_path:decl('function' if s=='def' else s,i+1)
            elif s=='package':
                q,_=_qualified(ts,i+1);add('namespace',q,n or t)
            elif s in ('import','export'):
                q,_=_qualified(ts,i+1);add('module_reference',q,t,import_style='scala_import')
        elif language=='groovy':
            if s in ('class','interface','trait','enum') and not preceded_path:decl(s,i+1)
            elif s=='package':
                q,_=_qualified(ts,i+1);add('namespace',q,n or t)
            elif s=='import':
                q,_=_qualified(ts,i+1);add('module_reference',q,t,import_style='groovy_import')
            elif s=='def' and _name(n):
                # `def x = ...` is a variable, `def f(` is a function.
                decl('function' if _at(ts,i+2,'(') else 'variable',i+1)
        elif language=='lua':
            if s=='function' and not preceded_path and _name(n):
                q,_=_qualified(ts,i+1,('.',':'))
                add('function',q,n)
            elif s=='local' and _at(ts,i+1,'function') and _name(ts[i+2] if i+2<len(ts) else None):
                decl('function',i+2)
            elif s=='require' and not preceded_path and n:
                got=_literal_arg(ts,i+1)
                if got:add('import',got[0],ts[got[1]],import_style='lua_require')
        elif language=='r':
            if _name(t) and n and n.text in ('<-','=') and _at(ts,i+2,'function') and _at(ts,i+3,'('):
                add('function',s,t)
            elif s=='source' and not preceded_path:
                got=_literal_arg(ts,i+1)
                if got:add('import',got[0],ts[got[1]],import_style='r_source')
        elif language=='julia':
            if s in ('module','baremodule','struct','function','macro','abstract','primitive') and not preceded_path:
                j=i+1+(s in ('abstract','primitive') and _at(ts,i+1,'type'))
                decl('function' if s=='function' else s,j)
            elif s=='mutable' and _at(ts,i+1,'struct'):decl('struct',i+2)
            elif s in ('using','import'):
                q,j=_qualified(ts,i+1)
                if q:add('module_reference',q,t,import_style='julia_import')
            elif s=='include' and not preceded_path:
                got=_literal_arg(ts,i+1)
                if got:add('import',got[0],ts[got[1]],import_style='julia_include')
        elif language=='perl':
            if s in ('package','sub') and _name(n):
                q,_=_qualified(ts,i+1,('::',))
                add('function' if s=='sub' else 'namespace',q,n)
            elif s in ('use','require') and n:
                if n.kind=='word':
                    q,j=_qualified(ts,i+1,('::',))
                    if q:add('module_reference',q,t,import_style='perl_module')
                elif n.kind=='string':add('import',n.text,n,import_style='perl_path')
        elif language=='haskell':
            if s=='module' and _name(n):
                q,_=_qualified(ts,i+1);add('namespace',q,n)
            elif s=='import':
                j=i+1+(n is not None and n.text=='qualified')
                q,_=_qualified(ts,j)
                if q:add('module_reference',q,t,import_style='haskell_import')
            elif s in ('data','newtype','type','class') and not preceded_path:decl(s,i+1)
            elif _name(t) and t.col==1 and (n and (n.text in ('::','=') or n.kind=='word')):
                # Layout-sensitive binding: the lexeme itself is not enough
                # for a sound function declaration. Capture only type sig.
                if _at(ts,i+1,'::'):add('signature',s,t)
        elif language=='erlang':
            if s=='-' and n and n.text in ('module','include','include_lib','export','import','record','define') and _at(ts,i+2,'('):
                key=n.text
                v=ts[i+3] if i+3<len(ts) else None
                if v and key=='module' and _name(v):add('namespace',v.text,v)
                elif v and key in ('include','include_lib') and v.kind=='string':add('import',v.text,v,import_style='erlang_include')
                elif v and key in ('import','record','define') and _name(v):add('module_reference' if key=='import' else key,v.text,v,import_style='erlang_import')
            elif _name(t) and (i==0 or p and p.text in ('.',';')) and _at(ts,i+1,'('):
                # Erlang function declaration: closing ) followed by ->/when.
                depth=0;j=i+1;commas=0;args=False
                while j<len(ts) and j<i+250:
                    if _at(ts,j,'('):depth+=1
                    elif _at(ts,j,')'):
                        depth-=1
                        if depth==0:break
                    elif depth==1 and _at(ts,j,','):commas+=1
                    elif depth==1 and not _at(ts,j,'('):args=True
                    j+=1
                if j<len(ts) and (_at(ts,j+1,'->') or _at(ts,j+1,'when')):
                    arity=(commas+1 if args else 0)
                    add('function',f'{s}/{arity}',t)
        elif language=='elixir':
            if s in ('defmodule','defprotocol','defimpl') and _name(n):
                q,_=_qualified(ts,i+1);add('module',q,n)
            elif s in ('def','defp','defmacro','defmacrop','defguard','defguardp') and _name(n):
                add('macro' if 'macro' in s or 'guard' in s else 'function',n.text,n)
            elif s in ('alias','import','require','use') and _name(n):
                q,_=_qualified(ts,i+1);add('module_reference',q,t,import_style='elixir_directive')
        elif language=='fsharp':
            if s in ('module','namespace') and _name(n):
                q,_=_qualified(ts,i+1);add('namespace',q,n)
            elif s=='open' and _name(n):
                q,_=_qualified(ts,i+1);add('module_reference',q,t,import_style='fsharp_open')
            elif s in ('type','member') and _name(n):decl(s,i+1)
            elif s=='let' and not preceded_path:
                j=i+1+(_at(ts,i+1,'rec') or _at(ts,i+1,'mutable'))
                if _name(ts[j] if j<len(ts) else None):decl('binding',j)
            elif s=='#' and _at(ts,i+1,'load') and i+2<len(ts) and ts[i+2].kind=='string':
                add('import',ts[i+2].text,ts[i+2],import_style='fsharp_load')
        elif language=='visual_basic':
            if s.lower() in ('class','interface','structure','enum','module','namespace','function','sub','property') and not preceded_path:
                # Avoid `End Class` and `End Function` closing constructs.
                if p and p.text.lower()=='end':continue
                if _name(n):add('function' if s.lower() in ('function','sub') else s.lower(),n.text,n)
            elif s.lower()=='imports' and _name(n):
                q,_=_qualified(ts,i+1)
                if q:add('module_reference',q,t,import_style='vb_imports')
        elif language=='zig':
            if s=='fn' and not preceded_path:decl('function',i+1)
            elif s in ('const','var') and _name(n):decl('binding',i+1)
            elif s=='@' and _at(ts,i+1,'import'):
                got=_literal_arg(ts,i+2)
                if got:add('import',got[0],ts[got[1]],import_style='zig_import')
        elif language=='shell':
            if s=='function' and _name(n):
                decl('function',i+1)
            elif _name(t) and (i==0 or p and (p.text in (';','}', '{') or p.line<t.line)):
                if _at(ts,i+1,'(') and _at(ts,i+2,')') and _at(ts,i+3,'{'):
                    add('function',s,t)
            if s in ('source','.') and (i==0 or p is None or p.line<t.line or p.text in (';','&&')):
                if n and n.kind=='string' and '$' not in n.text:
                    add('import',n.text,n,import_style='shell_source')
        elif language=='nim':
            if s in ('proc','func','method','iterator','template','macro','type','var','let','const') and not preceded_path:
                if _name(n):add('function' if s in ('proc','func','method','iterator') else s,n.text,n)
            elif s in ('import','include') and _name(n):
                q,_=_qualified(ts,i+1,('.',))
                if q:add('module_reference',q,t,import_style='nim_import')
            elif s=='from' and _name(n):
                q,_=_qualified(ts,i+1);add('module_reference',q,t,import_style='nim_from')
        if t.text=='{':brace_depth+=1
    seen=set();unique=[]
    for event in events:
        key=(event[0],event[1],event[2].offset,event[4])
        if key not in seen:seen.add(key);unique.append(event)
    return unique


def _candidate_import(source: str, value: str, style: str, known: set[str]) -> str | None:
    # Strictly literal, source-relative module paths. No arbitrary package
    # resolution, classpath probing, filesystem walking, or symlink following.
    rel=PurePosixPath(source).parent
    if not value or '\\' in value or value.startswith(('/', '~')) or '\x00' in value:
        return None
    if ':' in value or value.startswith(('..','@')) and style not in ('julia_include','fsharp_load','r_source','perl_path'):
        return None
    if style=='rust_mod_path':
        # #[path="..."] is a literal file path, not a module identifier.
        # Source-relative parent traversal is bounded by the inventory root.
        candidate=posixpath.normpath(str(rel/value))
        return (candidate if candidate not in ('','.','..') and
                not candidate.startswith(('../','/')) and candidate in known else None)
    suffix={
       'lua_require':'.lua','r_source':'.R','julia_include':'.jl','perl_path':'.pm',
       'erlang_include':'.hrl','fsharp_load':'.fs','rust_mod':'.rs',
       'zig_import':'.zig','dart_uri':'.dart','shell_source':'.sh'
    }.get(style)
    if not suffix:return None
    if style=='lua_require': value=value.replace('.','/')
    if style=='zig_import' and value in ('builtin','root','std'):
        return None
    if style=='dart_uri' and not value.startswith('.') and '/' not in value and not value.endswith('.dart'):
        return None
    if style=='rust_mod':
        # An external module declared in `src/foo.rs` lives under `src/foo/`,
        # whereas `src/lib.rs`, `src/main.rs` and `src/foo/mod.rs` use the
        # containing directory. Inline/attributed modules are not resolved.
        stem=PurePosixPath(source).stem
        base=rel if stem in ('lib','main','mod') else rel/stem
        options=[str(base / (value+'.rs')),str(base/value/'mod.rs')]
    else:
        if PurePosixPath(value).suffix:options=[str(rel/value)]
        else:options=[str(rel/(value+suffix))]
    if style=='r_source' and not PurePosixPath(value).suffix:
        options.extend([str(rel/(value+'.r'))])
    if style=='lua_require':options.append(str(rel/value/'init.lua'))
    matches=set()
    for candidate in options:
        candidate=posixpath.normpath(candidate)
        if candidate not in ('','.','..') and not candidate.startswith('../') and candidate in known:
            matches.add(candidate)
    return next(iter(matches)) if len(matches)==1 else None


def analyze_offline_multilanguage(root: Path,candidates: dict, _graph: dict,
                                  languages: tuple[str,...]) -> tuple[dict,list[dict]]:
    known={path for path, item in candidates.items() if item.get('language') in languages}
    root=root.resolve();graphs={};warnings=[];sources={};pending=[]
    for path in sorted(known):
        file=(root/path).resolve()
        if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size>MAX_BYTES:
            warnings.append({'path':path,'reason':'missing_outside_root_or_exceeds_size'});continue
        try:
            data=file.read_bytes(); text=data.decode('utf-8-sig')
            language=candidates[path]['language'];ts=tokenize(text,language)
            events=_declarations(ts,language)
        except (UnicodeError, OSError, ScanError) as exc:
            warnings.append({'path':path,'reason':'unparsed_source', 'detail':str(exc)[:100]});continue
        source_hash=sha256(data).hexdigest()
        info={'sha':source_hash,'lines':text.splitlines(),'language':language}
        sources[path]=info
        file_node={'id':_file_id(path),'path':path,'name':PurePosixPath(path).name,
                   'kind':'file','line':1,'confidence':'verified_file'}
        g={'nodes':[file_node],'edges':[],'hash':source_hash,
           'parser':f'{language}_source_tokenizer',
           'source_coordinates':'1_based_unicode_code_points',
           'caution':f'{language}: bounded lexical/structural declaration evidence; not full AST, type or call resolution.',
           'declaration_limitations':['lexical_structure_only','no_semantic_call_targets','no_compiler_validation']}
        graphs[path]=g
        def evidence(tok:Token):
            return {'path':path,'line':tok.line,'column':tok.col,'end_line':tok.end_line,
                    'end_column':tok.end_col,'source_sha256':source_hash,
                    'snippet':info['lines'][tok.line-1].strip()[:200] if tok.line<=len(info['lines']) else ''}
        for kind,name,tok,_other,style in events:
            if kind=='import':
                pending.append((path,name,tok,style))
                continue
            if kind=='module_reference':
                # Record syntax without claiming a local file target.
                kind='module_reference'
            sid=_node_id(path,kind+':'+name,tok.line,tok.col)
            node={'id':sid,'path':path,'name':name,'qualname':name,'kind':kind,
                  'line':tok.line,'column':tok.col,'end_line':tok.end_line,
                  'end_column':tok.end_col,'parent':file_node['id'],'confidence':'lexical_structure'}
            g['nodes'].append(node)
            g['edges'].append({'source':file_node['id'],'target':sid,
                'source_path':path,'target_path':path,'relation':'declares' if kind!='module_reference' else 'references',
                'resolution':'lexical_structure','analyzer':language+'-offline-structure',
                'evidence':evidence(tok)})
    # Explicit literal file references can be candidates only, never `calls`
    # or globally promoted `resolved_static` file-level edges.
    for path,value,tok,style in pending:
        g=graphs[path];info=sources[path]
        target=_candidate_import(path,value,style,set(graphs))
        sid=_node_id(path,'import:'+value,tok.line,tok.col)
        g['nodes'].append({'id':sid,'path':path,'name':value,'qualname':value,'kind':'module_reference',
                           'line':tok.line,'column':tok.col,'parent':_file_id(path),'confidence':'literal_syntax'})
        evid={'path':path,'line':tok.line,'column':tok.col,'end_line':tok.end_line,
              'end_column':tok.end_col,'source_sha256':info['sha'],
              'snippet':info['lines'][tok.line-1].strip()[:200] if tok.line<=len(info['lines']) else ''}
        g['edges'].append({'source':_file_id(path),'target':sid,'source_path':path,'target_path':path,
                           'relation':'references','resolution':'verified_literal_syntax',
                           'analyzer':info['language']+'-offline-structure','evidence':evid})
        if target:
            g['edges'].append({'source':sid,'target':_file_id(target),'source_path':path,'target_path':target,
                               'relation':'module_candidate','resolution':'candidate_import_path',
                               'analyzer':info['language']+'-offline-structure','evidence':evid})
    return graphs,warnings
