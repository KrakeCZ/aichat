"""Offline PHP version evidence, deliberately independent of the PHP runtime.

Never execute PHP, Composer, Docker or project code. Never scan outside the
explicit repository or follow symlinks. Manifest requirements are compatibility
claims, not evidence of the version deployed by a web server.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
from typing import Any

# Security notice is a PINNED reference snapshot, NOT an online scanner.
# Upstream fixes released 2026-10-21; vendor backports may differ.
PHP2SHELL_FIXED = {'8.2': '8.2.35', '8.3': '8.3.36', '8.4': '8.4.27', '8.5': '8.5.12'}
PHP_BRANCH_END = {'8.2': '2026-12-31', '8.3': '2027-12-31',
                  '8.4': '2028-12-31', '8.5': '2029-12-31'}
ADVISORY = 'https://www.php.net/archive/2026.php#2026-10-21-5'
MAX_MANIFEST_BYTES = 1_500_000
EXACT = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)(?:\+[^\s]+)?$')
BRANCH = re.compile(r'^v?(\d+)\.(\d+)$')
DOCKER = re.compile(r'^\s*FROM\s+php:(\d+\.\d+(?:\.\d+)?)(?:[-@\s]|$)', re.IGNORECASE)
TOOLS = re.compile(r'^\s*php\s+(\S+)\s*(?:#.*)?$')


def _allowed_file(root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        return None
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        return None
    file = root / path
    if not file.resolve().is_relative_to(root) or not file.is_file():
        return None
    if any((root.joinpath(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts)+1)):
        return None
    try:
        if file.stat().st_size > MAX_MANIFEST_BYTES:
            return None
    except OSError:
        return None
    return file


def _load(root: Path, path: str, issues: list) -> str | None:
    file = _allowed_file(root, path)
    if file is None:
        issues.append({'path':path, 'reason':'not_regular_in_scope_or_too_large'})
        return None
    try:
        return file.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError):
        issues.append({'path':path, 'reason':'unreadable_metadata'})
        return None


def _line(text: str, needle: str) -> int | None:
    # The file and raw value are authoritative evidence; the line is a hint
    # when duplicate JSON keys / values exist (never a compiler source map).
    for i, s in enumerate(text.splitlines(), 1):
        if needle in s:
            return i
    return None


def _evaluate(value: str, kind: str, today: date) -> dict:
    """Only classify EXACT pins. Do not pretend to implement Composer semver."""
    raw = value.strip()
    finding: dict[str, Any] = {'value':raw,'version_form':'constraint_not_resolved',
        'php2shell':'cannot_determine_from_range', 'branch_support':'not_determined',
        'message':'Požadavek není důkaz nasazené verze; kompletní Composer omezení zde nevyhodnocujeme.'}
    match = EXACT.fullmatch(raw)
    branch = None
    if match:
        version = tuple(int(x) for x in match.groups())
        branch = f'{version[0]}.{version[1]}'
        finding['version_form'] = 'exact_declared_pin'
        fixed = PHP2SHELL_FIXED.get(branch)
        if fixed is not None:
            minv = tuple(map(int, fixed.split('.')))
            finding['php2shell'] = ('below_upstream_fix' if version < minv else 'at_or_above_upstream_fix')
            finding['message'] = ('Přesná deklarovaná verze je pod upstream opravou.' if version < minv
                else 'Přesná deklarovaná verze dosahuje upstream opravy; provozní runtime není ověřen.')
        else:
            finding['php2shell'] = 'uncovered_or_legacy_branch'
            finding['message'] = 'Pro tuto řadu zde nemáme běžné upstream minimum; ověř podporu nebo vendor backport.'
    else:
        m = BRANCH.fullmatch(raw)
        if m:
            branch = f'{int(m.group(1))}.{int(m.group(2))}'
            finding['version_form'] = 'branch_without_patch'
            finding['message'] = 'Uvedena je pouze řada PHP, konkrétní patch ani nasazený runtime nejsou známé.'
        elif kind == 'composer_require_php':
            # Common, unambiguous legacy-only major/minor Composer ranges.
            # NEVER infer a deployed PHP runtime from such a constraint.
            legacy_only = (re.fullmatch(r'\^7\.\d+(?:\.\d+)?',raw)
                or re.fullmatch(r'~7\.\d+(?:\.\d+)?',raw)
                or re.fullmatch(r'~8\.[01](?:\.\d+)?',raw)
                or re.fullmatch(r'7\.\d+\.(?:\*|x)',raw,re.I)
                or re.fullmatch(r'8\.[01]\.(?:\*|x)',raw,re.I))
            finding['version_form'] = 'composer_constraint_not_resolved'
            if legacy_only:
                finding['branch_support'] = 'declared_range_limited_to_upstream_eol_branches'
                finding['message'] = ('Composer omezení je podle jednoduché syntaxe omezené na nepodporované PHP řady; '
                    'ověř modernizaci. Neprokazuje to aktuální verzi běžícího serveru.')
    if branch:
        expiry = PHP_BRANCH_END.get(branch)
        if expiry:
            finding['branch_support'] = 'eol' if today > date.fromisoformat(expiry) else 'supported_as_of_report_date'
            finding['branch_security_support_until'] = expiry
        elif int(branch.split('.')[0]) < 8 or (int(branch.split('.')[0])==8 and int(branch.split('.')[1]) < 2):
            finding['branch_support'] = 'upstream_eol'
        else:
            finding['branch_support'] = 'branch_not_in_frozen_support_table'
    return finding


def inspect_php_versions(root: Path, indexed_paths: set[str] | list[str], *, today: date | None = None) -> dict:
    """Read local manifests only. Zero subprocess/network calls by construction."""
    root = root.resolve()
    today = today or date.today()
    known = set(indexed_paths)
    php_files = sum(p.lower().endswith(('.php','.phtml','.php5')) for p in known)
    report: dict[str, Any] = {
        'schema':'atlas-php-version-evidence/1', 'report_date':today.isoformat(),
        'mode':'offline_read_only_no_php_or_composer_execution',
        'php_files_indexed':php_files,
        'runtime':{'inspected':False,'version':None,'note':'Runtime není kontrolován: žádné spuštění php -v, PHP-FPM ani Apache.'},
        'security_reference':{'name':'PHP2Shell','cves':['CVE-2026-109618','CVE-2026-109619'],
            'published':'2026-10-21', 'upstream_fixed_versions':PHP2SHELL_FIXED,
            'source':ADVISORY,
            'notice':'Statický referenční přehled. Neověřuje vendor backport, runtime nasazení ani další CVE.'},
        'evidence':[], 'review_items':[], 'unresolved':[],
        'conclusion':'Zdrojová metadata neumožňují spolehlivě určit nasazenou PHP verzi.'}
    # Only files discovered by Inventory and explicit sibling version metadata.
    manifests = sorted(p for p in known if p.endswith('composer.json') and (p=='composer.json' or p.endswith('/composer.json')))
    search_roots = {'.'} | {str(Path(p).parent).replace('\\','/') for p in manifests}
    requested: set[str] = set(manifests)
    for parent in search_roots:
        for name in ('composer.lock','.php-version','.tool-versions','Dockerfile'):
            requested.add((Path(parent) / name).as_posix())
    for path in sorted(requested):
        # Never probe the filesystem using a path controlled by a manipulated
        # source-index.json until traversal and symlink parents are ruled out.
        rel = Path(path)
        if (not path or rel.is_absolute() or '..' in rel.parts or '\\' in path or '\x00' in path):
            report['unresolved'].append({'path':path,'reason':'unsafe_metadata_path'})
            continue
        if any((root.joinpath(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
            report['unresolved'].append({'path':path,'reason':'not_regular_in_scope_or_too_large'})
            continue
        if not (root / rel).exists():
            continue
        text = _load(root,path,report['unresolved'])
        if text is None:
            continue
        entries = []
        if path.endswith(('composer.json','composer.lock')):
            try:
                data = json.loads(text)
                if not isinstance(data,dict):raise ValueError('root_not_object')
            except (ValueError,TypeError):
                report['unresolved'].append({'path':path,'reason':'invalid_json_metadata'});continue
            if path.endswith('composer.json'):
                require = data.get('require')
                if isinstance(require,dict) and isinstance(require.get('php'),str):
                    entries.append(('composer_require_php', require['php']))
                config = data.get('config')
                platform = config.get('platform') if isinstance(config,dict) else None
                if isinstance(platform,dict) and isinstance(platform.get('php'),str):
                    entries.append(('composer_config_platform_php',platform['php']))
            else:
                platform = data.get('platform')
                overrides = data.get('platform-overrides')
                if isinstance(platform,dict) and isinstance(platform.get('php'),str):
                    entries.append(('composer_lock_platform_php',platform['php']))
                if isinstance(overrides,dict) and isinstance(overrides.get('php'),str):
                    entries.append(('composer_lock_platform_override_php',overrides['php']))
        elif path.endswith('/Dockerfile') or path == 'Dockerfile':
            for i,line in enumerate(text.splitlines(),1):
                m=DOCKER.match(line)
                if m:entries.append(('docker_php_base_image',m.group(1),i))
        elif path.endswith('.php-version'):
            value = next((x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith('#')),'')
            if value:entries.append(('php_version_file',value))
        elif path.endswith('.tool-versions'):
            for i,line in enumerate(text.splitlines(),1):
                m=TOOLS.match(line)
                if m:entries.append(('asdf_php_version_hint',m.group(1),i))
        for record in entries:
            kind,value = record[:2]
            item = {'path':path,'kind':kind,'version':value,
                    'line_hint':record[2] if len(record)>2 else _line(text,value),
                    'evidence_type':'declared_configuration_not_deployed_runtime'}
            item.update(_evaluate(value,kind,today))
            report['evidence'].append(item)
            if item['php2shell']=='below_upstream_fix':
                report['review_items'].append({'path':path,'kind':kind,'version':value,
                    'reason':'declared_exact_version_below_upstream_php2shell_fix',
                    'message':'Zvaž aktualizaci deklarované verze; bez kontroly serveru nelze potvrdit zranitelnost nasazení.'})
            if item['branch_support'] in ('upstream_eol','eol','declared_range_limited_to_upstream_eol_branches'):
                report['review_items'].append({'path':path,'kind':kind,'version':value,
                    'reason':'declared_upstream_end_of_life',
                    'message':'Deklarované omezení odkazuje na řadu bez běžné upstream podpory; '
                              'vendor backport a skutečný server se kontrolují zvlášť.'})
    report['evidence'].sort(key=lambda x:(x['path'],x['kind'],x['version']))
    if php_files and not report['evidence']:
        report['unresolved'].append({'reason':'php_version_not_declared_in_scanned_metadata',
            'note':'PHP soubory byly rozpoznány, ale bez verzované konfigurace nelze určit požadovanou verzi.'})
    report['summary']={'declarations':len(report['evidence']),
         'review_items':len(report['review_items']), 'php_files':php_files,
         'unknown_project_php_version':not bool(report['evidence'])}
    return report
