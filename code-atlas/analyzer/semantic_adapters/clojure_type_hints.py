"""Read-only literal Clojure metadata tags; no inferred type or class loading."""
from __future__ import annotations
from .clojure_multimethod import _valid_target


def literal_type_tag(form):
    """Describe an explicit ^Type / ^{:tag Type}, never resolve its identity.

    Contradictory or computed tags are unavailable instead of guessing.
    Metadata is reader data: tagged literals and expressions are not run.
    """
    tags=[]
    for meta in form.metadata:
        tag=None
        if meta.kind=='atom' and _valid_target(meta) and meta.value not in {'true','false','nil'}:
            tag=meta
        elif meta.kind=='map' and len(meta.children)%2==0:
            for key,value in zip(meta.children[::2],meta.children[1::2]):
                if key.kind=='atom' and key.value==':tag':
                    if tag is not None or value.kind!='atom' or not _valid_target(value):
                        return None
                    tag=value
        if tag is not None:
            tags.append(tag)
    if len(tags)!=1:
        return None
    tag=tags[0]
    return {'source_spelling':tag.value,'resolution':'explicit_reader_metadata_only',
            'line':tag.line,'column':tag.col,
            'end_line':tag.end_line,'end_column':tag.end_col,
            'java_or_js_type_selected':False}
