"""Evidence-first PowerShell source analysis.

On machines with PowerShell this reads Parser.ParseFile AST (target is NOT
executed). Else a conservative lexical fallback finds command candidates;
lexical calls are NEVER described as semantically resolved. No program-specific
paths or function names occur here.
"""
from __future__ import annotations
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

FUNC = re.compile(r'\bfunction\s+([\w-]+)\b', re.I)
CLASS = re.compile(r'\bclass\s+([\w-]+)\b', re.I)
ASSIGN = re.compile(r'\$([A-Za-z_][\w]*)\s*(?:\+|\?|:)?=', re.I)
IDENT = re.compile(r'(?<![\w$.:\\-])([A-Za-z_][\w-]*)\b')
MAX_AST_TRANSPORT_BYTES = 16_000_000

EXCLUDE = {"function", "class", "param", "if", "elseif", "else", "switch", "while", "for", "foreach",
           "do", "until", "try", "catch", "finally", "return", "throw", "begin", "process", "end",
           "in", "break", "continue", "filter", "workflow", "using", "new-object", "import-module",
           "export-modulemember", "write-host", "write-output"}


def strip_comments_strings(text: str) -> list[str]:
    """Mask literals while keeping character offsets and exact line numbers.

    PowerShell's quoted strings and here-strings are not analysed as commands.
    The fallback does not attempt to fully parse nested $() interpolation.
    """
    out = []
    block = False
    here: str | None = None
    for line in text.splitlines():
        buf = list(line)
        quote: str | None = None
        i = 0
        while i < len(line):
            if here is not None:
                if line[i:].startswith(here) and line[:i].strip()=="":
                    here = None
                    for j in range(i,i+2):
                        if j<len(buf):buf[j]=" "
                    i+=2
                else:
                    buf[i]=" ";i+=1
                continue
            if block:
                if line.startswith('#>',i):
                    buf[i]=buf[i+1]=' ';block=False;i+=2
                else:buf[i]=' ';i+=1
                continue
            if quote:
                if line[i]=='`' and i+1<len(line):
                    buf[i]=buf[i+1]=' ';i+=2;continue
                if line[i]==quote:
                    quote=None
                buf[i]=' ';i+=1;continue
            if line.startswith('<#',i):
                block=True;buf[i]=buf[i+1]=' ';i+=2;continue
            if line.startswith('@"',i) or line.startswith("@'",i):
                here=line[i+1]+'@';buf[i]=buf[i+1]=' ';i+=2;continue
            if line[i]=='#':
                for j in range(i,len(buf)):buf[j]=' '
                break
            if line[i] in "\"'":
                quote=line[i];buf[i]=' ';i+=1;continue
            i+=1
        out.append(''.join(buf))
    return out


def lexical(text: str) -> dict:
    cleaned = strip_comments_strings(text)
    functions = []
    assignments=[]
    # A stack of known function bodies; brace counting is only a fallback,
    # so calls remain candidates even when the apparent owner is plausible.
    depth=0;active=[];pending=None
    commands=[]
    for number,line in enumerate(cleaned,1):
        stripped=line.strip()
        match=FUNC.search(line)
        if match and not line[:match.start()].strip().startswith(('$', '.')):
            name=match.group(1)
            functions.append({'name':name,'line':number,'end_line':number,'parameters':[]})
            pending=(name,number)
        # Owner at beginning of line, including a pending multiline definition.
        owner=active[-1][0] if active else (pending[0] if pending else None)
        for m in ASSIGN.finditer(line):assignments.append({'name':m.group(1),'line':number,'owner':owner})
        # Command positions: statement start, after { ; | &, or after parentheses.
        # In one-line functions, scan *after* the opening { as well.
        command_region = line
        command_offset = 0
        if match:
            open_at = line.find('{',match.end())
            if open_at < 0:
                command_region = ''
            else:
                command_offset = open_at + 1
                command_region = line[command_offset:]
                owner = match.group(1)
        if command_region.strip():
            for m in IDENT.finditer(command_region):
                word=m.group(1)
                if word.lower() in EXCLUDE:continue
                prefix=command_region[:m.start()]
                if prefix.strip() and not re.search(r'[;|{(&=(,]\s*$',prefix):continue
                if m.end()<len(command_region) and command_region[m.end()] in ':\\':continue
                commands.append({'name':word,'line':number,'owner':owner,'snippet':text.splitlines()[number-1].strip()[:200]})
        openings=line.count('{');closings=line.count('}')
        if pending and openings:
            active.append((pending[0],depth+1,pending[1]));pending=None
        depth+=openings-closings
        while active and depth<active[-1][1]:
            name,_,start=active.pop()
            for f in functions:
                if f['name']==name and f['line']==start:
                    f['end_line']=number;break
        if pending and number-pending[1]>8:pending=None
    return {'parser':'powershell_lexical','errors':[], 'functions':functions,'commands':commands,
            'assignments':assignments, 'caution':'Lexical candidate calls only; no PowerShell AST available.'}


@lru_cache(maxsize=1024)
def analyze_file(filename: str, mtime_ns: int, size: int) -> dict:
    path=Path(filename)
    source=path.read_text(encoding='utf-8-sig',errors='replace')
    executable=shutil.which('pwsh') or shutil.which('powershell')
    if executable:
        script=Path(__file__).with_name('powershell_ast_dump.ps1')
        command=[executable,'-NoLogo','-NoProfile','-NonInteractive']
        if Path(executable).name.lower().startswith('powershell'):
            command+=['-ExecutionPolicy','Bypass']
        command+=['-File',str(script),'-InputPath',str(path)]
        try:
            # Windows PowerShell 5.1 can emit console-code-page text rather
            # than UTF-8. Reading that stream with subprocess(text=True)
            # raises in a background _readerthread on Python 3.14 and loses
            # the AST. Our helper instead produces ASCII Base64(UTF-8 JSON).
            # Capture bytes to avoid any implicit host-locale decoding.
            proc=subprocess.run(command,capture_output=True,timeout=25,check=True)
            output=proc.stdout.strip()
            marker=b'ATLAS_PS_AST_UTF8_B64:'
            if (len(output)<=MAX_AST_TRANSPORT_BYTES and
                    output.startswith(marker) and b'\n' not in output and b'\r' not in output):
                payload=base64.b64decode(output[len(marker):],validate=True)
                if len(payload)>MAX_AST_TRANSPORT_BYTES:
                    raise ValueError('oversized decoded PowerShell AST')
                raw=json.loads(payload.decode('utf-8-sig'))
                if (isinstance(raw,dict) and raw.get('parser')=='powershell_ast' and
                        all(isinstance(raw.get(k),list) for k in
                            ('functions','commands','assignments','errors'))):
                    # Genuine AST data are separate from lexical candidates.
                    return raw
            fallback_reason='powershell_ast_transport_unavailable'
        except (subprocess.SubprocessError,OSError,ValueError,TypeError,
                UnicodeError,binascii.Error):
            fallback_reason='powershell_ast_transport_unavailable'
        # Never make a parse failure look like a successfully verified AST.
        fallback=lexical(source)
        fallback['ast_fallback_reason']=fallback_reason
        return fallback
    return lexical(source)


def analyze_path(path: Path) -> dict:
    info=path.stat()
    return analyze_file(str(path.resolve()),info.st_mtime_ns,info.st_size)
