"""Read-only reconciliation of *externally supplied* Clojure compiler observations.

No Clojure, ClojureScript compiler, project macro, build script or dependency is
executed. The report is supplied separately by the user, is NOT authenticated,
and is never sufficient to assert compiler_binding_confirmed=True. It can only
corroborate an independently discovered, source-anchored candidate Var edge.

Strict project root, source hash, classpath scenario, platform, CST token,
declaration identity, and edge identity checks prevent stale/cross-project
reports from silently changing Atlas navigation. Failing checks are diagnostics,
not compiler evidence.
"""
from __future__ import annotations

from hashlib import sha256
import base64
import binascii
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

SCHEMA = 'atlas-clojure-external-compiler-report/1'
SIGNED_SCHEMA = 'atlas-clojure-external-compiler-report/2'
TRUST_SCHEMA = 'atlas-clojure-trusted-report-producer-key/1'
SIGNING_DOMAIN = b'UNIVERSAL-CODE-ATLAS/CLOJURE-REPORT-V2/ED25519\0'

MAX_REPORT_BYTES = 1_000_000
MAX_REPORT_FILES = 256
MAX_REPORT_OBSERVATIONS = 2500
MAX_OBSERVATIONS_PER_SOURCE = 250
MAX_PRODUCER_LENGTH = 100
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_SYMBOL = re.compile(r'[^\s\[\]{}()";,]+\Z')
_ALLOWED_RELATIONS = {
    'call': frozenset({
        'clojure_call_candidate', 'clojure_local_var_reference_candidate',
        'clojure_referred_call_candidate', 'clojure_macro_phase_candidate',
        'clojure_macro_invocation_candidate', 'clojure_referred_macro_phase_candidate',
        'clojure_referred_macro_invocation_candidate'}),
    'value': frozenset({'clojure_var_value_reference_candidate'}),
}


class ReportInvalid(ValueError):
    """Failure category is fixed; never expose local paths or report contents."""


def _object_no_duplicates(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ReportInvalid('duplicate_json_member')
        result[key] = value
    return result


def _shape(value, required, label):
    if type(value) is not dict or set(value) != set(required):
        raise ReportInvalid('invalid_' + label + '_shape')


def _positive(value, label):
    if type(value) is not int or not (1 <= value <= 2_000_000):
        raise ReportInvalid('invalid_' + label + '_coordinate')


def _utf8_scalar(value):
    """Reject lone JSON surrogate escapes (not Unicode scalar values).

    Python's JSON parser intentionally permits them, but a later SHA-256
    encoding, JSON graph write, or terminal transport must never crash Atlas.
    Do not normalize valid Unicode: the Clojure reader is source-exact.
    """
    if type(value) is not str:
        return False
    try:
        value.encode('utf-8', errors='strict')
        return True
    except UnicodeEncodeError:
        return False


def _digest(value):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ReportInvalid('invalid_sha256')


def _path(value):
    if (not _utf8_scalar(value) or not value or len(value) > 240 or
            "\\" in value or '\x00' in value or
            any(ord(char) < 32 for char in value)):
        raise ReportInvalid('invalid_project_relative_path')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ('..', '.') for part in value.split('/')):
        raise ReportInvalid('invalid_project_relative_path')
    return value


def _read_snapshot(path_value, *, limit, failure):
    """Bounded, read-only, symlink-averse snapshot of a supplied file.

    Check every path component, not merely its basename. O_NOFOLLOW stops
    replacement of the leaf with a symlink between lstat and open where
    supported; fstat rejects FIFOs, devices, directories, and concurrent
    modification. The result is a *byte snapshot*, never executable code.
    """
    try:
        path = Path(path_value).expanduser().absolute()
        if '..' in Path(path_value).parts:
            raise ReportInvalid(failure)
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ReportInvalid(failure)
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        flags |= getattr(os, 'O_BINARY', 0)
        if (os.name == 'posix' and hasattr(os, 'O_NOFOLLOW') and
                hasattr(os, 'O_DIRECTORY') and os.open in os.supports_dir_fd):
            # Open every directory by file descriptor with O_NOFOLLOW, so a
            # concurrent symlink replacement in an ANCESTOR cannot redirect
            # this read after our initial path-component inspection.
            dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            parent_fd = os.open(path.anchor, dir_flags)
            try:
                for part in path.parts[1:-1]:
                    next_fd = os.open(part, dir_flags, dir_fd=parent_fd)
                    os.close(parent_fd)
                    parent_fd = next_fd
                fd = os.open(path.name, flags, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
        else:
            # Windows has no portable O_NOFOLLOW+dir_fd. Explicit reparse
            # checks remain a best-effort guard, NOT a sandbox boundary.
            if any(getattr(part, 'is_junction', lambda: False)()
                   for part in (path, *path.parents)):
                raise ReportInvalid(failure)
            fd = os.open(path, flags)
        try:
            initial = os.fstat(fd)
            if not stat.S_ISREG(initial.st_mode) or initial.st_size > limit:
                raise ReportInvalid(failure)
            with os.fdopen(fd, 'rb', closefd=False) as opened:
                data = opened.read(limit + 1)
            final = os.fstat(fd)
            if (len(data) > limit or initial.st_size != len(data) or
                    (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns,
                     initial.st_ctime_ns) !=
                    (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns,
                     final.st_ctime_ns)):
                raise ReportInvalid(failure)
            return data
        finally:
            os.close(fd)
    except (OSError, ValueError, TypeError, OverflowError):
        raise ReportInvalid(failure) from None


def _scenario(options):
    from .clojure_source_context import normalize_cli_aliases
    try:
        aliases = list(normalize_cli_aliases(options.get('_atlas_clojure_cli_aliases', [])))
    except ValueError:
        raise ReportInvalid('invalid_current_aliases') from None
    candidate = options.get('_atlas_clojure_classpath_snapshot')
    digest = (None if candidate is None else
              sha256(_read_snapshot(candidate, limit=2_000_000,
                                    failure='current_classpath_snapshot_not_readable')).hexdigest())
    return {'cli_aliases': aliases, 'classpath_snapshot_sha256': digest}


def _signed_scenario(options, files):
    """Bound v2 report to *all* indexed Clojure source bytes, not just cited sites.

    This is an input fingerprint, NOT evidence of successful compiler analysis,
    an effective build environment or an authenticated project origin.
    """
    baseline = _scenario(options)
    indexed = sorted((path, file['sha']) for path, file in files.items())
    encoded = json.dumps(indexed, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    target = options.get('_atlas_clojure_target_jvm')
    if target is not None and (type(target) is not int or not 8 <= target <= 27):
        raise ReportInvalid('invalid_current_target_jvm')
    return dict(baseline, target_jvm=target,
                source_inventory_sha256=sha256(encoded).hexdigest())


def _decode_base64(value, byte_size, reason):
    if type(value) is not str or len(value) > 160 or not value.isascii():
        raise ReportInvalid(reason)
    try:
        encoded = value.encode('ascii')
        output = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ReportInvalid(reason) from None
    if len(output) != byte_size or base64.b64encode(output) != encoded:
        raise ReportInvalid(reason)
    return output


def _verified_producer(raw_report, report, options):
    """Verify user-pinned Ed25519 producer provenance over exact report bytes.

    Authenticity of a signer's *key* is a user trust decision. Signature alone
    cannot demonstrate that the signer ran a compiler or isolated project macros.
    """
    key_file = options.get('_atlas_clojure_compiler_trust_key')
    sig_file = options.get('_atlas_clojure_compiler_report_signature')
    if not key_file or not sig_file:
        raise ReportInvalid('signed_report_requires_key_and_signature')
    raw_key = _read_snapshot(key_file, limit=2048, failure='trusted_key_unreadable_or_unsafe')
    raw_sig = _read_snapshot(sig_file, limit=256, failure='report_signature_unreadable_or_unsafe')
    try:
        key_data = json.loads(raw_key.decode('utf-8'), object_pairs_hook=_object_no_duplicates,
                              parse_int=_bounded_json_int, parse_float=_reject_json_float,
                              parse_constant=lambda _: (_ for _ in ()).throw(ReportInvalid('nonfinite_number')))
    except ReportInvalid:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ReportInvalid('trusted_key_not_valid_json') from None
    _shape(key_data, ('schema','producer','ed25519_public_key_base64'), 'trusted_key')
    if key_data['schema'] != TRUST_SCHEMA:
        raise ReportInvalid('unsupported_trusted_key_schema')
    _shape(key_data['producer'], ('name','version'), 'trusted_key_producer')
    if key_data['producer'] != report['producer']:
        raise ReportInvalid('trusted_producer_identity_mismatch')
    key_bytes = _decode_base64(key_data['ed25519_public_key_base64'], 32, 'invalid_ed25519_public_key')
    try:
        sig_text = raw_sig.decode('ascii')
    except UnicodeError:
        raise ReportInvalid('invalid_ed25519_signature') from None
    signature = _decode_base64(sig_text.removesuffix('\r\n').removesuffix('\n'), 64, 'invalid_ed25519_signature')
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise ReportInvalid('ed25519_crypto_backend_not_installed') from None
    try:
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, SIGNING_DOMAIN + raw_report)
    except (InvalidSignature, ValueError):
        raise ReportInvalid('report_signature_verification_failed') from None
    return {'method':'ed25519_detached_raw_v2', 'public_key_sha256':sha256(key_bytes).hexdigest(),
            'producer':dict(key_data['producer']), 'compiler_execution_attested_by_atlas':False}


def _bounded_json_int(digits):
    # Coordinates have a 2,000,000 upper bound; oversized integer text must
    # be rejected *before* CPython's version-dependent int parser limits.
    # This also avoids a digit-count CPU hazard on older Python versions.
    if len(digits) > 12:
        raise ReportInvalid('json_integer_literal_limit')
    return int(digits)


def _reject_json_float(_literal):
    # The schema contains only integer coordinates and SHA/string/null fields.
    raise ReportInvalid('json_noninteger_number_not_supported')


def _load(report_path: str, scenario: dict, *, schema=SCHEMA):
    raw = _read_snapshot(report_path, limit=MAX_REPORT_BYTES,
                         failure='report_missing_symlink_or_size_limit')
    try:
        report = json.loads(raw.decode('utf-8'), object_pairs_hook=_object_no_duplicates,
                            parse_int=_bounded_json_int, parse_float=_reject_json_float,
                            parse_constant=lambda _: (_ for _ in ()).throw(ReportInvalid('nonfinite_number')))
    except ReportInvalid:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ReportInvalid('report_not_valid_json') from None
    _shape(report, ('schema','producer','scenario','sources'), 'report')
    if report['schema'] != schema:
        raise ReportInvalid('unsupported_report_schema')
    _shape(report['producer'], ('name','version'), 'producer')
    for value in report['producer'].values():
        if not _utf8_scalar(value) or not value or len(value) > MAX_PRODUCER_LENGTH or any(ord(c)<32 for c in value):
            raise ReportInvalid('invalid_producer_claim')
    _shape(report['scenario'], ('cli_aliases','classpath_snapshot_sha256') if schema == SCHEMA else
           ('cli_aliases','classpath_snapshot_sha256','target_jvm','source_inventory_sha256'), 'scenario')
    if schema == SIGNED_SCHEMA:
        target_jvm = report['scenario']['target_jvm']
        if target_jvm is not None and (type(target_jvm) is not int or not 8 <= target_jvm <= 27):
            raise ReportInvalid('invalid_signed_target_jvm')
        _digest(report['scenario']['source_inventory_sha256'])
    aliases = report['scenario']['cli_aliases']
    from .clojure_source_context import normalize_cli_aliases
    try:
        normalized = list(normalize_cli_aliases(aliases))
    except ValueError:
        raise ReportInvalid('invalid_report_aliases') from None
    if type(aliases) is not list or normalized != aliases:
        raise ReportInvalid('invalid_report_aliases')
    if report['scenario'] != scenario:
        raise ReportInvalid('analysis_scenario_mismatch')
    sources = report['sources']
    if type(sources) is not list or len(sources) > MAX_REPORT_FILES:
        raise ReportInvalid('report_file_limit')
    count = 0
    unique_sites = set()
    for source in sources:
        _shape(source, ('path','sha256','platform','observations'), 'source')
        _path(source['path']); _digest(source['sha256'])
        if source['platform'] not in ('clj','cljs'):
            raise ReportInvalid('unsupported_report_platform')
        items = source['observations']
        if type(items) is not list or len(items) > MAX_OBSERVATIONS_PER_SOURCE:
            raise ReportInvalid('report_source_observation_limit')
        count += len(items)
        if count > MAX_REPORT_OBSERVATIONS:
            raise ReportInvalid('report_observation_limit')
        for item in items:
            _shape(item, ('category','symbol','site','target'), 'observation')
            if type(item['category']) is not str or item['category'] not in _ALLOWED_RELATIONS:
                raise ReportInvalid('unsupported_observation_category')
            if (not _utf8_scalar(item['symbol']) or not item['symbol'] or len(item['symbol']) > 240 or
                    any(ord(c) < 32 for c in item['symbol']) or
                    _SYMBOL.fullmatch(item['symbol']) is None):
                raise ReportInvalid('invalid_observed_symbol')
            _shape(item['site'], ('line','column','end_line','end_column'), 'site')
            for key, value in item['site'].items():
                _positive(value, key)
            site = item['site']
            if (site['end_line'], site['end_column']) <= (site['line'], site['column']):
                raise ReportInvalid('empty_or_inverted_site')
            _shape(item['target'], ('path','sha256','qualified_var'), 'target')
            _path(item['target']['path']); _digest(item['target']['sha256'])
            qualified = item['target']['qualified_var']
            if (not _utf8_scalar(qualified) or len(qualified) > 240 or qualified.count('/') != 1 or
                    any(not segment for segment in qualified.split('/')) or _SYMBOL.fullmatch(qualified) is None):
                raise ReportInvalid('invalid_target_qualified_var')
            identity = (source['path'], source['platform'],site['line'],site['column'],
                        site['end_line'],site['end_column'],item['category'])
            if identity in unique_sites:
                raise ReportInvalid('duplicate_observation_site')
            unique_sites.add(identity)
    return report,sha256(raw).hexdigest(),raw


def _safe_cst_atom_locations(cst):
    """Index nonquoted physical atoms ONCE, not once per report observation.

    The exported CST is bounded, but repeated O(nodes) searches can still be
    quadratic on a large codebase. Mark each subtree from an opaque/discard
    prefix as unsafe and cache the result for its descendants.
    """
    if cst.get('status') != 'complete_source_coverage_structural_subset':
        return set()
    nodes = cst.get('nodes', [])
    by_id = {node.get('id'):node for node in nodes}
    unsafe_cache = {}

    def unsafe(node_id):
        if node_id is None:
            return False
        if node_id in unsafe_cache:
            return unsafe_cache[node_id]
        trail = []
        current = node_id
        while current is not None and current not in unsafe_cache:
            item = by_id.get(current)
            if item is None:
                # Invalid/unresolved ownership blocks all claims.
                for item_id in trail:
                    unsafe_cache[item_id] = True
                return True
            trail.append(current)
            if item.get('category') in ('reader_discard','opaque_reader_prefix'):
                for item_id in trail:
                    unsafe_cache[item_id] = True
                return True
            current = item.get('parent')
        verdict = unsafe_cache.get(current, False)
        for item_id in trail:
            unsafe_cache[item_id] = verdict
        return verdict

    return {
        (node.get('line'),node.get('column'),node.get('end_line'),node.get('end_column'),
         node.get('text_sha256'))
        for node in nodes
        if node.get('kind') == 'token' and node.get('category') == 'atom' and
           not unsafe(node.get('parent'))
    }


def _site_matches_cst(atom_locations, observed, symbol):
    return (observed['line'],observed['column'],observed['end_line'],observed['end_column'],
            sha256(symbol.encode('utf-8')).hexdigest()) in atom_locations


def reconcile_compiler_report(graphs: dict, files: dict, options: dict, root: Path):
    """Annotate existing candidate edges only, or return bounded diagnostics.

    This boundary deliberately cannot create or promote a call/reference edge.
    It can be extended by an independently trusted compiler integration later.
    """
    selected = options.get('_atlas_clojure_compiler_report')
    signed_requested = bool(options.get('_atlas_clojure_compiler_trust_key') or
                            options.get('_atlas_clojure_compiler_report_signature'))
    for graph in graphs.values():
        graph['external_compiler_report_evidence'] = {'status': 'not_requested'}
    if not selected and not signed_requested:
        return
    try:
        if not selected:
            raise ReportInvalid('signed_report_requires_report')
        if signed_requested:
            if (not options.get('_atlas_clojure_compiler_trust_key') or
                    not options.get('_atlas_clojure_compiler_report_signature')):
                raise ReportInvalid('signed_report_requires_key_and_signature')
            report, report_sha, raw = _load(selected, _signed_scenario(options, files), schema=SIGNED_SCHEMA)
            authentication = _verified_producer(raw, report, options)
        else:
            report, report_sha, raw = _load(selected, _scenario(options))
            authentication = None
        mapped = {}
        seen_sources = set()
        for source in report['sources']:
            path = source['path']; platform = source['platform']
            if (path,platform) in seen_sources:
                raise ReportInvalid('duplicate_source_platform')
            seen_sources.add((path,platform))
            record = files.get(path)
            if record is None or path not in graphs or record['sha'] != source['sha256']:
                raise ReportInvalid('source_missing_or_sha_mismatch')
            allowed = {'clj'} if path.endswith('.clj') else {'cljs'} if path.endswith('.cljs') else (
                {'clj','cljs'} if path.endswith('.cljc') else set())
            if platform not in allowed:
                raise ReportInvalid('source_platform_mismatch')
            mapped.setdefault(path, []).append(source)
        # Validate all target identities/hashed files before annotating ANY edge.
        for sources in mapped.values():
            for source in sources:
                for obs in source['observations']:
                    target = obs['target']
                    entry = files.get(target['path'])
                    if entry is None or entry['sha'] != target['sha256'] or target['path'] not in graphs:
                        raise ReportInvalid('target_missing_or_sha_mismatch')
        prepared = {}
        nodes_by_id = {
            node['id']:node
            for target_graph in graphs.values() for node in target_graph['nodes']
            if node.get('kind') in ('function','macro','variable','multimethod','protocol','record','type','interface')
        }
        for path, sources in mapped.items():
            edges = graphs[path]['edges']
            safe_tokens = _safe_cst_atom_locations(files[path]['source_cst'])
            lookup = {}
            for edge in edges:
                evidence = edge.get('evidence') or {}
                node = nodes_by_id.get(edge.get('target'))
                if not node or edge.get('source_path') != path:
                    continue
                for category, relations in _ALLOWED_RELATIONS.items():
                    if edge.get('relation') not in relations:
                        continue
                    key = (category,edge.get('target_path'),node.get('qualname'),
                           evidence.get('line'),evidence.get('column'),
                           evidence.get('end_line'),evidence.get('end_column'))
                    lookup.setdefault(key,[]).append(edge)
            accepted = []
            conflicts = []
            for source in sources:
                for obs in source['observations']:
                    site=obs['site']; target=obs['target']
                    if not _site_matches_cst(safe_tokens,site,obs['symbol']):
                        reason='site_not_active_exact_cst_atom'
                        matches=[]
                    else:
                        key = (obs['category'], target['path'], target['qualified_var'],
                               site['line'],site['column'],site['end_line'],site['end_column'])
                        matches = lookup.get(key, [])
                        reason = 'no_matching_independent_source_candidate' if not matches else 'ambiguous_independent_source_candidate'
                    if len(matches) != 1:
                        if len(conflicts) < 32:
                            conflicts.append({'line':site['line'],'column':site['column'],
                                              'platform':source['platform'],'reason':reason})
                        continue
                    edge=matches[0]
                    # For .cljc edges scoped merely to 'shared' cannot prove
                    # either platform. Never cross-assign those observations.
                    if edge.get('source_platform') != source['platform']:
                        if len(conflicts) < 32:
                            conflicts.append({'line':site['line'],'column':site['column'],
                                              'platform':source['platform'],'reason':'edge_platform_not_exact'})
                        continue
                    accepted.append((edge, source['platform']))
            prepared[path] = (sources, accepted, conflicts)
        for path,(sources,accepted,conflicts) in prepared.items():
            for edge,platform in accepted:
                edge['external_compiler_report'] = {
                    'status':'externally_reported_candidate_agrees_not_authenticated',
                    'report_sha256':report_sha,'observed_platform':platform,
                    'compiler_executed_by_atlas':False, 'compiler_binding_confirmed':False,
                    'producer_signature_verified':bool(authentication)}
            graphs[path]['external_compiler_report_evidence']={
                'status':'report_reconciled_against_existing_candidate_only',
                'report_sha256':report_sha,'claimed_producer':report['producer'],
                'observations':sum(len(s['observations']) for s in sources),
                'matched_candidates':len(accepted),
                'unmatched_or_conflicting':sum(len(s['observations']) for s in sources)-len(accepted),
                'conflict_examples':conflicts,
                'compiler_execution_by_atlas':False,
                'report_producer_authenticated':bool(authentication),
                'signature_evidence':authentication,
                'compiler_binding_confirmed':False}
    except ReportInvalid as exc:
        # Fail closed for the entire report. Never export user-provided path,
        # snippets or arbitrary exception messages.
        for graph in graphs.values():
            graph['external_compiler_report_evidence'] = {
                'status':'unavailable', 'reason':str(exc),
                'compiler_execution_by_atlas':False,
                'compiler_binding_confirmed':False}
