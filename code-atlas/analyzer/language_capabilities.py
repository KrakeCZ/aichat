"""Honest, repo-independent coverage manifest for the current adapters.

Identification != parsing != semantic resolution. A language may be recognized
and still have zero safe semantic relations. Never advertise 28 'deep parsers'
based on extension matching.
"""
from __future__ import annotations
from collections import Counter
from typing import Any

# The languages are established language identifiers from Inventory, not names
# of applications/files/modules. This is a roadmap + honest capability index.
TARGET_LANGUAGES = (
    'python','javascript','typescript','java','kotlin','c','cpp','csharp','rust','go',
    'swift','objective_c','dart','ruby','php','scala','lua','powershell','shell',
    'r','julia','perl','haskell','elixir','erlang','clojure','fsharp','visual_basic',
    'sql','groovy','zig','nim'
)
LANGUAGE_PARSERS = {
    'python':('python_ast','ast_symbols_and_selected_resolved_calls'),
    'powershell':('powershell_ast_when_available_or_limited_fallback','selected_calls_and_static_or_candidate_module_imports'),
    'javascript':('typescript_compiler_ast_if_node_available_or_lexical','static_imports_and_select_verified_calls'),
    'typescript':('typescript_compiler_ast_if_node_available_or_lexical','static_imports_and_select_verified_calls'),
    'rust':('rust_readonly_source_structure','lexical_declarations_and_literal_local_mod_candidates_not_type_resolved'),
    'go':('go_stdlib_ast_if_available_or_legacy','AST_declarations_with_go_types_resolved_direct_calls_when_complete_and_candidate_calls_otherwise'),
    'shell':('shell_readonly_source_structure','lexical_functions_and_literal_source_candidates'),
    'kotlin':('kotlin_compiler_psi_if_kotlinc_available_or_declarations_fallback','PSI_syntax_plus_optional_local_JVM_compiler_resolved_calls_when_compile_succeeds'),
    'java':('javac_compiler_ast_if_JDK_available','AST_declarations_and_selected_symbol_resolved_calls'),
    'c':('clang_compiler_ast_if_available','AST_declarations_and_selected_direct_call_references'),
    'cpp':('clang_compiler_ast_if_available','AST_declarations_and_selected_direct_call_references'),
    'csharp':('csharp_readonly_source_structure','lexical_declarations_and_using_directives_not_type_resolved'),
    'swift':('swift_compiler_ast_if_swiftc_available','AST_declarations_and_select_compiler_bound_calls'),
    'objective_c':('clang_objc_ast_if_available','AST_declarations_and_methods_without_dynamic_dispatch_guarantees'),
    'php':('php_builtin_token_parser_if_local_CLI_available','syntax_validated_tokens_and_literal_include_candidates_no_type_binding'),
    'ruby':('ruby_ripper_ast_if_available_or_legacy','AST_declarations_and_limited_require_relative_candidates'),
    'dart':('dart_readonly_source_structure','lexical_declarations_and_literal_import_candidates_not_type_resolved'),
    'clojure':('builtin_clojure_structural_forms_subset','literal_namespace_and_top_level_definitions_with_candidates_no_macro_expansion_or_static_binding'),
    'sql':('builtin_sql_structural_subset_lexer','CREATE_TABLE_VIEW_and_selected_statement_structure_with_candidate_unique_local_table_references_no_dialect_validation'),
    'xml':('builtin_expat_xml_1_0_namespace_reader','verified_element_hierarchy_and_attribute_values_without_DTD_schema_or_external_resolution'),
    'scala':('scala_readonly_source_structure','lexical_declarations_and_import_directives'),
    'groovy':('groovy_readonly_source_structure','lexical_declarations_and_import_directives'),
    'lua':('lua_readonly_source_structure','lexical_functions_and_literal_require_candidates'),
    'r':('r_readonly_source_structure','lexical_function_bindings_and_source_candidates'),
    'julia':('julia_readonly_source_structure','lexical_functions_modules_and_literal_include_candidates'),
    'perl':('perl_readonly_source_structure','lexical_packages_subroutines_and_literal_require_candidates'),
    'haskell':('haskell_readonly_source_structure','lexical_module_imports_and_type_declarations'),
    'erlang':('erlang_readonly_source_structure','lexical_module_attributes_and_function_heads'),
    'elixir':('elixir_readonly_source_structure','lexical_module_function_macro_directives'),
    'fsharp':('fsharp_readonly_source_structure','lexical_bindings_modules_and_open_directives'),
    'visual_basic':('visual_basic_readonly_source_structure','lexical_type_and_member_declarations'),
    'zig':('zig_readonly_source_structure','lexical_functions_and_literal_import_candidates'),
    'nim':('nim_readonly_source_structure','lexical_functions_and_module_directives'),

}

# These levels describe actual *observed* parser runs, not promised support.
# Every level is partial: even an AST+references index cannot prove runtime
# behavior, dependency injection, reflection or every source-line side effect.
PARSER_LEVELS = {
    'python_ast': 'partial_semantic',
    'typescript_compiler_ast': 'partial_semantic',
    'javac_ast': 'partial_semantic',
    'clang_ast': 'partial_semantic',
    'powershell_ast': 'syntax_ast',
    'kotlin_psi': 'syntax_ast',
    'go_stdlib_ast': 'syntax_ast',
    'go_stdlib_ast_typed': 'partial_semantic',
    'ruby_ripper_ast': 'syntax_ast',
    'php_tokenizer_validated': 'syntax_tokens',
    'sql_structural_subset': 'syntax_subset',
    'xml_expat_source_structure': 'syntax_ast',
    'shell_source_tokenizer': 'lexical_structure',
    'rust_source_tokenizer': 'lexical_structure',
    'csharp_source_tokenizer': 'lexical_structure',
    'dart_source_tokenizer': 'lexical_structure',
    'scala_source_tokenizer': 'lexical_structure',
    'groovy_source_tokenizer': 'lexical_structure',
    'lua_source_tokenizer': 'lexical_structure',
    'r_source_tokenizer': 'lexical_structure',
    'julia_source_tokenizer': 'lexical_structure',
    'perl_source_tokenizer': 'lexical_structure',
    'haskell_source_tokenizer': 'lexical_structure',
    'erlang_source_tokenizer': 'lexical_structure',
    'elixir_source_tokenizer': 'lexical_structure',
    'fsharp_source_tokenizer': 'lexical_structure',
    'visual_basic_source_tokenizer': 'lexical_structure',
    'zig_source_tokenizer': 'lexical_structure',
    'nim_source_tokenizer': 'lexical_structure',

    'clojure_structural_forms_subset': 'syntax_subset',
    'swift_compiler_ast': 'partial_semantic',
    'clang_objc_ast': 'syntax_ast',
    'powershell_lexical': 'declarations_only',
    'javascript_lexical': 'declarations_only',
    'declarations_only': 'declarations_only',
    'file_only': 'file_only',
}


def capabilities(graph_data: dict[str, Any], symbol_index: dict|None=None) -> dict[str, Any]:
    detected=Counter(n.get('properties',{}).get('language','unknown') for n in graph_data.get('nodes',[]) if n.get('type')=='file')
    symbols_by_language=Counter()
    parsers_by_language={}
    statuses_by_language={}
    overlays_by_language={}
    if symbol_index:
        path_lang={n['id']:n.get('properties',{}).get('language','unknown') for n in graph_data['nodes'] if n.get('type')=='file'}
        for path,entry in symbol_index.get('files',{}).items():
            lang=path_lang.get(path,'unknown')
            symbols_by_language[lang]+=entry.get('symbol_count',0)
            parsers_by_language.setdefault(lang,Counter())[entry.get('parser','unknown')]+=1
            statuses_by_language.setdefault(lang,Counter())[entry.get('source_parse_status','unknown')]+=1
            if entry.get('semantic_overlay'):
                overlays_by_language.setdefault(lang, Counter())[entry['semantic_overlay']] += 1
    node_lang={n['id']:n.get('properties',{}).get('language','unknown') for n in graph_data['nodes'] if n.get('type')=='file'}
    edges_by_language=Counter()
    for edge in graph_data.get('edges',[]):
        source=edge.get('source');
        if source in node_lang:edges_by_language[node_lang[source]]+=1
    known=set(TARGET_LANGUAGES)|set(detected)
    result=[]
    for lang in sorted(known):
        method,scope=LANGUAGE_PARSERS.get(lang,('file_recognition','no_semantic_claim'))
        parser_counts = parsers_by_language.get(lang, Counter())
        observed_levels = Counter()
        for parser, count in parser_counts.items():
            observed_levels[PARSER_LEVELS.get(parser, 'unknown_parser')] += count
        result.append({'language':lang,'files_detected':detected[lang], 'method':method,
                       'guarantee':scope,'symbol_count':symbols_by_language[lang],
                       'observed_analysis_levels':dict(sorted(observed_levels.items())),
                       'files_with_syntax_ast':sum(count for parser,count in parser_counts.items()
                                                  if PARSER_LEVELS.get(parser) in ('syntax_ast','partial_semantic')),
                       'files_with_lexical_structure':sum(count for parser,count in parser_counts.items() if PARSER_LEVELS.get(parser)=='lexical_structure'),
                       'files_with_validated_tokens':sum(count for parser,count in parser_counts.items()
                                                          if PARSER_LEVELS.get(parser)=='syntax_tokens'),
                       'files_with_partial_semantics':sum(count for parser,count in parser_counts.items()
                                                          if PARSER_LEVELS.get(parser)=='partial_semantic'),
                       'file_edges':edges_by_language[lang],
                       'observed_symbol_parsers':dict(sorted(parsers_by_language.get(lang,{}).items())),
                       'semantic_overlays': dict(sorted(overlays_by_language.get(lang, {}).items())),
                       'files_with_rejected_or_unavailable_source':statuses_by_language.get(lang,Counter()).get('rejected_or_unavailable',0),
                       'files_with_only_heuristic_or_unavailable_analysis':statuses_by_language.get(lang,Counter()).get('heuristic_or_unavailable',0),
                       'files_with_parsed_partial_source':statuses_by_language.get(lang,Counter()).get('parsed_partial',0),
                       'semantically_complete':False})
    return {'schema':'atlas-language-capabilities-1',
      'target_language_count':len(TARGET_LANGUAGES), 'identified_target_languages':sum(detected.get(l,0)>0 for l in TARGET_LANGUAGES),
      'note':'32 target languages for incremental support; observed_analysis_levels measure actual parsers. AST parsing is not semantic comprehension of every line.',
      'languages':result}
