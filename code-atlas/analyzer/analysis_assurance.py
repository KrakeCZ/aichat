"""Conservative release-independent quality gate for source analysis output.

This report distinguishes a structurally valid graph from a trustworthy, fully
resolved semantic model. It NEVER claims forensic or semantic completeness based
on file recognition, balanced delimiters, AST syntax or passing unit tests.

The report uses only already generated offline indices; no project execution,
network I/O, compiler invocation or recursive filesystem discovery.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from analyzer.language_capabilities import PARSER_LEVELS

SCHEMA = 'atlas-analysis-assurance/1'
SPEC_CHECKED_AS_OF = '2026-11-04'

# Reference release observed at clojure.org/releases/downloads on 2026-11-04.
# This describes the specification review only: no compiler/runtime binary is
# bundled, invoked or verified; the actual project version is NOT inferred.
REFERENCE_RELEASES = {
    'clojure': {
        'upstream_stable_jvm_release': '1.12.6',
        'release_date': '2026-09-02',
        'project_version_detected': False,
        'reference_runtime_differential_tests_passed': False,
        'clojurescript_reference_version': 'not_verified',
    },
}

# Official specifications are relevant to the claimed language constructs,
# NOT proof that our partial parser implements them completely.
REFERENCES = {
    'clojure': [
        'https://clojure.org/reference/reader',
        'https://clojure.org/reference/namespaces',
        'https://clojure.org/reference/libs',
    ],
}

# All remain unproven until the *complete* language-specific reference and
# dynamic/build-context comparison suite is available and has been executed.
REQUIRED_GATES = (
    'official_grammar_and_version_conformance',
    'reader_and_macro_semantics',
    'scopes_bindings_and_type_resolution',
    'imports_and_build_variant_resolution',
    'control_and_data_flow',
    'cross_file_call_and_dynamic_dispatch_validation',
    'differential_reference_toolchain_tests',
    'comprehensive_negative_and_fuzz_testing',
    'evidence_chain_and_input_snapshot_integrity',
    'independent_security_and_forensic_audit',
)


def build_assurance(
    source_index: dict[str, Any],
    symbols: dict[str, Any],
    graph_integrity: dict[str, Any],
    symbol_integrity: dict[str, Any],
    graph_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct an honest, bounded assurance report for the current run."""
    source_files = {
        item['path']: item
        for item in source_index.get('files', [])
        if isinstance(item, dict) and isinstance(item.get('path'), str)
    }
    evidence_issues = defaultdict(list)
    for issue in symbol_integrity.get('problems', []):
        if isinstance(issue, dict):
            evidence_issues[issue.get('file', '')].append(issue.get('problem', 'unknown'))
    adapter_issues = defaultdict(list)
    global_issues = []
    for issue in symbols.get('errors', []):
        if not isinstance(issue, dict):
            continue
        # Diagnostics are bounded so a malicious source cannot inflate the report.
        label = {'reason': str(issue.get('reason', 'unknown'))[:120],
                 'details': str(issue.get('details', ''))[:200]}
        path = issue.get('path')
        if isinstance(path, str) and path in source_files:
            if len(adapter_issues[path]) < 10:
                adapter_issues[path].append(label)
        elif len(global_issues) < 40:
            global_issues.append(label)
    files = {}
    levels = Counter()
    languages = Counter()
    parse_states = Counter()
    unsupported_files = 0
    unresolved_total = 0
    unresolved_files = 0
    for path, item in sorted(source_files.items()):
        result = symbols.get('files', {}).get(path, {})
        parser = str(result.get('parser', 'not_analyzed'))
        level = PARSER_LEVELS.get(parser, 'unknown_or_file_only')
        if parser == 'not_analyzed':
            level = 'file_only'
        language = str(item.get('language', 'unknown'))
        source_parse_status = result.get('source_parse_status','unknown')
        parse_states[source_parse_status] += 1
        limitations = result.get('reader_limitations', [])
        # Never accept these as interpreted results, only as bounded diagnostics.
        if not isinstance(limitations, list):
            limitations = []
        if limitations:
            unsupported_files += 1
        # Observed unresolved qualified references are not evidence of absent
        # dependencies. They must remain visible to downstream users and AI.
        unresolved_count = result.get('unresolved_reference_count', 0)
        if not isinstance(unresolved_count, int) or unresolved_count < 0:
            unresolved_count = 0
        unresolved_total += unresolved_count
        unresolved_files += bool(unresolved_count)
        levels[level] += 1
        languages[language] += 1
        files[path] = {
            'language': language,
            'parser': parser,
            'source_parse_status': source_parse_status,
            'source_parse_issue': result.get('source_parse_issue'),
            'observed_level': level,
            'semantic_completeness': 'not_certified',
            'platform_scope': result.get('platform_scope', 'not_applicable'),
            'source_sha256': result.get('source_sha256', ''),
            'analyzer_warnings': adapter_issues.get(path, []),
            'unsupported_reader_constructs': limitations[:30],
            'lexical_scope_limitations': result.get('lexical_scope_limitations', []),
            'lexical_scope_suppression_count': result.get('lexical_scope_suppression_count', 0),
            'qualified_scope_suppression_count': result.get('qualified_scope_suppression_count', 0),
            'qualified_scope_suppression_examples': result.get('qualified_scope_suppression_examples', [])[:20],
            'unresolved_reference_count': unresolved_count,
            'unresolved_references': (result.get('unresolved_references', []) or [])[:40],
            'evidence_problems': evidence_issues.get(path, [])[:20],
        }
    integrity_ok = bool(graph_integrity.get('valid')) and bool(symbol_integrity.get('valid'))
    input_metadata = (graph_data or {}).get('metadata', {})
    large_files_count = int(input_metadata.get('omitted_large_source_count', 0))
    large_files = input_metadata.get('omitted_large_sources', [])
    if not isinstance(large_files, list):
        large_files=[]
    # Candidate relations must NEVER silently become resolved static calls.
    # Applies to all language adapters and the exported global graph.
    candidate_promotion_violations = []
    for edge in (graph_data or {}).get('edges', []):
        relation = str(edge.get('type', ''))
        confidence = str(edge.get('confidence_v3', ''))
        if (relation.endswith('_candidate') or relation == 'call_candidate') and confidence != 'candidate':
            if len(candidate_promotion_violations) < 100:
                candidate_promotion_violations.append({
                    'source': str(edge.get('source', ''))[:160],
                    'target': str(edge.get('target', ''))[:160],
                    'relation': relation[:120], 'confidence': confidence[:80],
                })
    return {
        'schema': SCHEMA,
        'specification_review_date': SPEC_CHECKED_AS_OF,
        'scope': 'Only selected repository and already-generated offline indices',
        'verdict': 'NOT_CERTIFIED_FOR_COMPLETE_SEMANTIC_OR_FORENSIC_ANALYSIS',
        'forensic_certified': False,
        'all_relationships_proven': False,
        'structural_integrity_passed': integrity_ok,
        'candidate_classification_passed': not candidate_promotion_violations,
        'candidate_promotion_violations': candidate_promotion_violations,
        'structural_integrity_is_not_semantic_correctness': True,
        'summary': {
            'files_detected': len(source_files),
            'files_omitted_due_to_inventory_size_limit':large_files_count,
            'all_source_files_indexed':large_files_count == 0,
            'languages_detected': dict(sorted(languages.items())),
            'parser_levels': dict(sorted(levels.items())),
            'source_parse_status_counts': dict(sorted(parse_states.items())),
            'files_with_unresolved_reader_constructs': unsupported_files,
            'files_with_unresolved_references': unresolved_files,
            'unresolved_reference_total': unresolved_total,
            'files_with_adapter_warnings': sum(bool(adapter_issues.get(p)) for p in source_files),
            'verified_graph_structure': integrity_ok,
            'candidate_promotion_violations': len(candidate_promotion_violations),
        },
        'required_completion_gates': {gate:'not_certified' for gate in REQUIRED_GATES},
        'official_references_by_language': {name: urls for name, urls in REFERENCES.items()
                                             if languages.get(name, 0)},
        'official_reference_releases': {name: item for name, item in REFERENCE_RELEASES.items()
                                         if languages.get(name, 0)},
        'candidate_links_are_not_proven_calls': True,
        'global_analyzer_warnings': global_issues,
        'files_omitted_due_to_inventory_size_limit':large_files[:1000],
        'files': files,
        'note': (
            'This is an explicit non-certification report. A valid graph and correct file hashes'
            ' do not establish complete parsing, correct dispatch, runtime behavior or actual project semantics.'
        ),
    }
