"""Local PHP token parser: validated syntax + literal include candidates, no code runs.

Requires a local PHP CLI with the standard tokenizer extension. Invokes only the
TRUSTED AtlasPhpTokenBridge.php script under `php -n`; PHP files from the
repository are read as DATA with token_get_all(TOKEN_PARSE). No Composer,
include, eval, project scripts, dependency download, or network calls.

Importantly: PHP tokens are not a typed AST or runtime call graph. The index
advertises parser='php_tokenizer_validated', not 'php_ast'.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from analyzer.symbol_graph import _file_id, _node_id

BRIDGE = Path(__file__).with_name('AtlasPhpTokenBridge.php')
MAX_FILES = 300
MAX_FILE_BYTES = 1_000_000
MAX_STDOUT = 15_000_000
MAX_TOTAL_BYTES = 16_000_000
# Upstream fixed releases published 21 Oct 2026 for PHP2Shell:
# CVE-2026-109618, CVE-2026-109619. No trust in an unknown local binary.
MINIMUM_PHP_PATCH = {(8, 2): 35, (8, 3): 36, (8, 4): 27, (8, 5): 12}
SECURITY_ADVISORY = 'https://www.php.net/archive/2026.php#2026-10-21-5'


def _version_is_approved(output: str) -> tuple[bool, str]:
    """Conservative upstream version gate, not a full distro patch attestation."""
    match = re.search(r'^PHP (\d+)\.(\d+)\.(\d+)([^\r\n]*)', output, re.MULTILINE)
    if not match:
        return False, 'unrecognized_PHP_version'
    major, minor, patch = (int(match.group(i)) for i in (1, 2, 3))
    tail = match.group(4)
    if re.search(r'(?:alpha|beta|rc|dev|snapshot)', tail, re.I):
        return False, 'prerelease_PHP_version'
    if '(cli)' not in tail.lower():
        return False, 'not_PHP_CLI'
    if (major, minor) in MINIMUM_PHP_PATCH:
        ok = patch >= MINIMUM_PHP_PATCH[(major, minor)]
    else:
        # Unknown major versions are not silently assumed safe; 8.6 is still
        # prerelease as of the 2026-11-04 audit.
        ok = False
    return ok, f'{major}.{minor}.{patch}'


def _safe_php_environment() -> dict[str, str]:
    """No inherited PHP_INI, LD_PRELOAD, auto-start, proxy or composer settings."""
    safe = ('SystemRoot', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'TMPDIR', 'LANG', 'LC_ALL')
    return {key: value for key, value in os.environ.items() if key in safe}


def _php_binary() -> str | None:
    specified = os.environ.get('ATLAS_PHP_CLI')
    if specified:
        path = Path(specified).expanduser()
        return str(path) if path.is_absolute() and path.is_file() else None
    # A PATH-resolved PHP is allowed only after explicit opt-in.
    return shutil.which('php')


def _secure_php_args(executable: str) -> list[str]:
    env = _safe_php_environment()
    version = subprocess.run([executable, '-n', '-v'], cwd=tempfile.gettempdir(),
                             env=env, capture_output=True, text=True, timeout=6)
    approved, version_detail = _version_is_approved(version.stdout)
    if version.returncode or not approved:
        raise ValueError(f'php_runtime_unapproved:{version_detail}')
    return _php_args(executable)



def _php_args(executable: str) -> list[str]:
    """Disable PHP.ini; load only the tokenizer if it's a dynamic extension.

    No arbitrary PHP.ini, auto-prepend hooks, Composer autoloader, or remote I/O.
    """
    args = [executable, '-n', '-d', 'allow_url_fopen=0', '-d', 'allow_url_include=0']
    check = subprocess.run(args + ['-r', 'echo extension_loaded("tokenizer") ? "yes" : "no";'],
                           cwd=tempfile.gettempdir(), env=_safe_php_environment(),
                           capture_output=True, text=True, timeout=8)
    if check.returncode != 0:
        raise RuntimeError('PHP CLI probe failed')
    if check.stdout.strip() != 'yes':
        raise RuntimeError('PHP tokenizer is not built in under -n; no optional extension will be loaded')
    return args


def analyze_php(root: Path, candidates: dict, _graph: dict) -> tuple[dict, list[dict]]:
    paths = sorted(p for p, v in candidates.items() if v.get('language') == 'php')
    if not paths:
        return {}, []
    if os.environ.get('ATLAS_ENABLE_PHP_TOKENIZER') != '1':
        return {}, [{'reason': 'php_tokenizer_opt_in_required',
                     'details': 'For local PHP execution opt in with ATLAS_ENABLE_PHP_TOKENIZER=1; no external PHP process was started.'}]
    executable = _php_binary()
    if not executable:
        return {}, [{'reason': 'php_cli_missing', 'details': 'Optional explicitly approved PHP CLI is not installed.'}]
    try:
        args = _secure_php_args(executable)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {}, [{'reason': 'php_runtime_security_blocked', 'details': str(exc)[:150],
                     'advisory': SECURITY_ADVISORY, 'code_execution': 'none_project_source'}]
    root = root.resolve()
    warnings = []
    chosen = []
    source_data = {}
    inputs = []
    total_bytes = 0
    for path in paths:
        if len(chosen) >= MAX_FILES:
            warnings.append({'reason': 'php_file_limit', 'limit': MAX_FILES})
            break
        lexical = root / path
        file = lexical.resolve()
        # No symbolic links at any path component, even if they point inside root.
        parts = lexical.relative_to(root).parts if lexical.is_relative_to(root) else ()
        if (not parts or any((root.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts)+1))
                or not file.is_relative_to(root) or not file.is_file() or file.stat().st_size > MAX_FILE_BYTES):
            warnings.append({'path': path, 'reason': 'php_source_skipped_outside_root_or_size'})
            continue
        raw = file.read_bytes()
        if total_bytes + len(raw) > MAX_TOTAL_BYTES:
            warnings.append({'path':path, 'reason':'php_total_source_size_limit', 'limit':MAX_TOTAL_BYTES})
            break
        total_bytes += len(raw)
        chosen.append(path)
        # Transfer source as *data via stdin*. PHP helper never opens project files.
        inputs.append({'path':path, 'content_b64':base64.b64encode(raw).decode('ascii')})
        source_data[path] = {
            'hash': hashlib.sha256(raw).hexdigest(),
            'lines': raw.decode('utf-8-sig', errors='replace').splitlines(),
        }
    if not chosen:
        return {}, warnings
    try:
        with tempfile.TemporaryDirectory(prefix='atlas-php-tokens-') as tmp:
            # Trusted script only. Runtime and paths are bounded; not an OS sandbox.
            restrictions = [
                '-d', 'memory_limit=192M',
                '-d', 'max_execution_time=30',
                '-d', 'open_basedir=' + os.pathsep.join((str(BRIDGE.parent.resolve()), tmp)),
                '-d', 'disable_functions=exec,passthru,shell_exec,system,proc_open,popen,pcntl_exec,curl_exec,fsockopen,stream_socket_client',
                '-d', 'opcache.enable_cli=0',
                '-d', 'ffi.enable=0',
                '-d', 'file_uploads=0',
                '-d', 'session.upload_progress.enabled=0',
            ]
            if BRIDGE.is_symlink() or not BRIDGE.is_file():
                raise RuntimeError('untrusted_php_bridge_location')
            proc = subprocess.run(args + restrictions + [str(BRIDGE)], cwd=tmp,
                                  input=json.dumps({'files':inputs}).encode('utf8'),
                                  env=_safe_php_environment(),
                                  capture_output=True, timeout=35)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode('utf-8', errors='replace')[:250])
            if len(proc.stdout) > MAX_STDOUT:
                raise RuntimeError('php_token_output_limit')
            records = json.loads(proc.stdout)
            if (not isinstance(records, list) or len(records) > len(chosen)
                or any(not isinstance(rec, dict) for rec in records)):
                raise RuntimeError('invalid_php_token_records')
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return {}, warnings + [{'reason': 'php_token_bridge_failed', 'details': str(exc)[:220]}]

    result = {}
    includes = {}
    def evidence(path, line, column=1):
        info = source_data[path]
        return {
            'path': path, 'line': line, 'column': column,
            'source_sha256': info['hash'],
            'snippet': info['lines'][line - 1].strip()[:200] if 0 < line <= len(info['lines']) else '',
        }

    for rec in records:
        path = rec.get('path')
        if not isinstance(path,str) or path not in source_data:
            continue
        if rec.get('error'):
            warnings.append({'reason': 'php_parse_error', 'path': path, 'details': rec.get('detail','')[:180]})
            continue
        file_node = {'id': _file_id(path), 'path': path, 'name': Path(path).name,
                     'kind': 'file', 'line': 1, 'confidence': 'verified_file'}
        graph = {'nodes': [file_node], 'edges': [], 'parser': 'php_tokenizer_validated',
                 'hash': source_data[path]['hash'],
                 'caution': ('PHP native token_get_all(TOKEN_PARSE) validates syntax and declaration tokens; '
                             'this is NOT a typed AST, a resolved namespace reference or proof of runtime includes.')}
        nodes_by_qual = {}
        pending_parents = []
        for rec_node in rec.get('nodes', []):
            if not isinstance(rec_node, dict):
                continue
            line = rec_node.get('line')
            col = rec_node.get('column')
            if not isinstance(line,int) or not 1 <= line <= len(source_data[path]['lines']):
                continue
            if not isinstance(col,int) or col < 1:
                continue
            name, qual, kind = (rec_node.get('name'), rec_node.get('qual'), rec_node.get('kind'))
            if (not all(isinstance(x,str) and x for x in (name,qual,kind))
                or len(name) > 512 or len(qual) > 1024
                or kind not in ('class','interface','trait','enum','function','method')):
                continue
            node = {'id': _node_id(path,qual,line,col),'path': path,'name': name,
                    'qualname':qual, 'kind':kind,'line':line,'column':col,'confidence':'verified_syntax'}
            graph['nodes'].append(node)
            nodes_by_qual.setdefault(qual, []).append(node)
            pending_parents.append((node,rec_node.get('parent','')))
        for node, parent_qual in pending_parents:
            options = nodes_by_qual.get(parent_qual,[]) if parent_qual else [file_node]
            if len(options) != 1:
                options = [file_node]  # avoid fabricating containment for ambiguous declarations
            parent = options[0]
            node['parent'] = parent['id']
            graph['edges'].append({
                'source': parent['id'],'target': node['id'],
                'source_path':path,'target_path':path,
                'relation':'declares','resolution':'verified_syntax',
                'analyzer':'php-tokenizer','evidence':evidence(path,node['line'],node['column']),
            })
        result[path] = graph
        includes[path] = rec.get('includes', [])

    for path, candidates_include in includes.items():
        graph = result[path]
        source_id = _file_id(path)
        seen = set()
        for item in candidates_include:
            if not isinstance(item, dict):
                continue
            value = item.get('value','')
            if not isinstance(value,str) or not value or '\x00' in value or '\\' in value:
                continue
            # __DIR__ + '/relative' uses the including file's directory.
            # A plain literal may depend on include_path/CWD; still only a candidate.
            if value.startswith('/') and item.get('base') != '__DIR__':
                continue
            if len(value) > 400 or '://' in value or value.startswith(('php:', 'data:')):
                continue
            base = root / Path(path).parent
            target = (base / value.lstrip('/') if item.get('base') == '__DIR__' else base / value).resolve()
            if not target.is_relative_to(root) or target == root:
                continue
            destination = target.relative_to(root).as_posix()
            if destination not in result or destination == path:
                continue
            line,col = item.get('line'), item.get('column',1)
            if not isinstance(line,int) or not 1 <= line <= len(source_data[path]['lines']):
                continue
            if not isinstance(col,int) or col < 1:
                continue
            if (destination,line) in seen:
                continue
            seen.add((destination,line))
            target_id = _file_id(destination)
            if target_id not in {n['id'] for n in graph['nodes']}:
                graph['nodes'].append({**result[destination]['nodes'][0], 'external':True})
            graph['edges'].append({
                'source':source_id,'target':target_id,
                'source_path':path,'target_path':destination,
                'relation':'include_candidate','resolution':'php_literal_include_candidate',
                'analyzer':'php-tokenizer','evidence':evidence(path,line,col),
            })
    return result, warnings
