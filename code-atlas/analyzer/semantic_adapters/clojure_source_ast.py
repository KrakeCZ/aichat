"""Non-executing, bounded Clojure/CLJS form AST with exact source ranges.

The structural reader creates a distinct node for every returned form and its
reader metadata. Excluded reader-discarded forms remain represented in the
separate lossless CST. Platform-selected .cljc forms are separate AST views.
No macroexpansion or type resolution is implied. Unsupported reader dispatches
stay explicitly opaque, never silently converted to function calls.
"""
from __future__ import annotations
from bisect import bisect_right
from hashlib import sha256
import json

MAX_AST_NODES = 60000
MAX_AST_DEPTH = 96
MAX_AST_JSON_BYTES = 6_000_000


def source_form_ast(text, forms, sha, platform):
    if platform not in {'clj','cljs','shared'}:
        return {'status':'unavailable','reason':'unknown_platform'}
    starts=[0]
    for i,c in enumerate(text):
        if c=='\n':starts.append(i+1)
    utf8=[0]
    utf16=[0]
    for c in text:
        utf8.append(utf8[-1]+len(c.encode('utf-8')))
        utf16.append(utf16[-1]+(2 if ord(c)>0xFFFF else 1))
    nodes=[]
    errors=[]
    def visit(form, parent, role, depth):
        if depth>MAX_AST_DEPTH or len(nodes)>=MAX_AST_NODES:
            raise ValueError('source_ast_node_or_depth_limit')
        if not (isinstance(form.line,int) and isinstance(form.col,int) and
                isinstance(form.end_line,int) and isinstance(form.end_col,int) and
                1<=form.line<=len(starts) and 1<=form.end_line<=len(starts)):
            raise ValueError('source_ast_missing_source_coordinate')
        begin=starts[form.line-1]+form.col-1
        end=starts[form.end_line-1]+form.end_col-1
        if not (0<=begin<end<=len(text)):
            raise ValueError('source_ast_invalid_source_range')
        ident=len(nodes)
        record={'id':ident,'parent':parent,'role':role,'kind':form.kind,
                'line':form.line,'column':form.col,
                'end_line':form.end_line,'end_column':form.end_col,
                'source_range':[begin,end],
                'utf8_byte_range':[utf8[begin],utf8[end]],
                'utf16_codeunit_range':[utf16[begin],utf16[end]],
                'source_sha256':sha}
        if form.kind=='atom':
            record['symbol_or_literal_spelling']=form.value
            original_spelling=text[begin:end]
            if original_spelling != form.value:
                # A selected namespaced-map key is a derived reader value,
                # not the bytes originally present at the source location.
                record['spelling_is_reader_derived']=True
                record['source_spelling_sha256']=sha256(original_spelling.encode('utf-8')).hexdigest()
        elif form.kind=='namespaced_map':
            record['reader_namespace_prefix']=form.value
            record['reader_semantics']='map_keys_need_namespace_context'
        elif form.kind=='map' and form.value.startswith('#:'):
            record['reader_namespace_prefix']=form.value
            record['reader_semantics']='map_keys_statistically_qualified'
        elif form.kind=='opaque':
            record['reader_semantics']='opaque_not_resolved'
            record['reader_prefix']=form.value[:64]
        elif form.kind in {'string','char','regex'}:
            record['literal_value_sha256']=sha256(form.value.encode('utf-8')).hexdigest()
            record['reader_semantics']='source_literal_no_evaluation'
        else:
            record['reader_semantics']='syntactic_children_only'
        nodes.append(record)
        for j,m in enumerate(form.metadata):
            visit(m,ident,'reader_metadata:'+str(j),depth+1)
        for j,child in enumerate(form.children):
            # Structural AST role is NOT execution evidence. In particular a
            # quoted list head remains syntax, never a resolved call target.
            if form.kind in ('map','namespaced_map'):
                # Before reader-conditional selection, #?@ can insert an
                # unknown number of key/value forms. No key parity is proven.
                uncertain = any(x.kind=='opaque' and x.value in ('#?', '#?@')
                                for x in form.children)
                role = ('map_unprojected_entry:'+str(j) if uncertain else
                        ('map_key:' if j%2==0 else 'map_value:')+str(j//2))
            elif form.kind=='list' and j==0:
                role='list_head_syntax'
            elif form.kind=='opaque':
                role='reader_operand:'+str(j)
            else:
                role='collection_element:'+str(j)
            visit(child,ident,role,depth+1)
        return ident
    try:
        roots=[]
        for top in forms:
            roots.append(visit(top,None,'top_level',0))
        # A form child must lie in its parent span; metadata may occur before
        # the decorated form, so is attached by role without fabricated spans.
        for entry in nodes:
            if entry['parent'] is None or entry['role'].startswith('reader_metadata:'):
                continue
            parent=nodes[entry['parent']]
            if not (parent['source_range'][0]<=entry['source_range'][0] and
                    entry['source_range'][1]<=parent['source_range'][1]):
                raise ValueError('source_ast_child_outside_parent_range')
        result={'status':'source_form_ast_structural_subset',
            'source_sha256':sha,
            'decoded_source_sha256':sha256(text.encode('utf-8')).hexdigest(),
            'platform':platform,'coordinates':'one_based_unicode_end_exclusive_and_utf8_bytes',
            'reader_execution':False,'macroexpansion':False,
            'decoded_utf16_codeunit_length':utf16[-1],
            'opaque_reader_nodes':sum(node['kind']=='opaque' for node in nodes),
            'node_count':len(nodes),'top_level_form_count':len(roots),
            'roots':roots,'nodes':nodes,
            'discarded_forms':'preserved_in_clojure_source_cst_only'}
        if len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode('utf-8'))>MAX_AST_JSON_BYTES:
            raise ValueError('source_ast_export_byte_limit')
        return result
    except (ValueError,RecursionError,UnicodeEncodeError) as exc:
        return {'status':'unavailable','reason':str(exc)[:120],
                'source_sha256':sha,'platform':platform}
