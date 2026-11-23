"""Read-only, bounded XML 1.0 well-formedness / element hierarchy for Atlas.

No DTD, entity resolver, schema download, stylesheets, or network. XML nodes
are *source structure*, not type/URI/IDREF resolution, not evaluated values.
Files that are not safely parseable have no semantic graph; the shared Atlas
export reports the diagnostic and keeps the source file entry only.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from xml.parsers import expat

from analyzer.symbol_graph import _file_id, _node_id

MAX_XML_BYTES = 2_000_000
MAX_XML_NODES = 50_000
MAX_XML_DEPTH = 128
MAX_XML_ATTRIBUTES = 512
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
XML_ID = "{" + XML_NAMESPACE + "}id"


class XMLReaderUnavailable(ValueError):
    pass


def _xml_name(name: str) -> str:
    # Expat's namespace mode uses the requested '}' separator between URI
    # and local name; retain the URI explicitly, never resolve by prefix guess.
    return '{'+name if '}' in name else name


def _expanded_name(qname: str, bindings: dict[str, str], *, attribute: bool) -> str:
    """Namespace expanded name; default namespace never applies to attributes."""
    if ':' in qname:
        prefix, local = qname.split(':', 1)
        uri = bindings.get(prefix)
        if not uri:
            raise XMLReaderUnavailable('xml_qname_prefix_not_bound')
        return '{' + uri + '}' + local
    if not attribute and bindings.get(''):
        return '{' + bindings[''] + '}' + qname
    return qname


def _valid_ncname(value: str) -> bool:
    """XML 1.0 Fifth Edition NCName code-point ranges, excluding ':'."""
    if not value:
        return False

    def start(c: str) -> bool:
        n = ord(c)
        return (c == '_' or 'A' <= c <= 'Z' or 'a' <= c <= 'z'
                or any(lo <= n <= hi for lo, hi in (
                    (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x2FF),
                    (0x370, 0x37D), (0x37F, 0x1FFF), (0x200C, 0x200D),
                    (0x2070, 0x218F), (0x2C00, 0x2FEF), (0x3001, 0xD7FF),
                    (0xF900, 0xFDCF), (0xFDF0, 0xFFFD),
                    (0x10000, 0xEFFFF))))

    def char(c: str) -> bool:
        n = ord(c)
        return (start(c) or c in '-.' or '0' <= c <= '9'
                or n == 0xB7 or 0x300 <= n <= 0x36F
                or 0x203F <= n <= 0x2040)

    return start(value[0]) and all(char(c) for c in value[1:])


def _normalized_xml_id(value: str) -> str:
    # XML 1.0 ID whitespace: only XML whitespace, not arbitrary Unicode spaces.
    return re.sub(r'[ \t\r\n]+', ' ', value).strip(' ')


def _allowed(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(p in ('.','..') for p in rel.parts):
        raise XMLReaderUnavailable('invalid_source_path')
    lexical = root.joinpath(*rel.parts)
    # Reject symlinks at *every* component, even if their target remains in
    # the repo. An Atlas inventory alone is not a security boundary.
    check = lexical
    while check != root:
        if check.is_symlink():
            raise XMLReaderUnavailable('symlink_source_path')
        check = check.parent
    file = lexical.resolve(strict=True)
    if not file.is_relative_to(root) or not file.is_file():
        raise XMLReaderUnavailable('outside_root_or_not_file')
    return file


def _physical_utf8_enabled(data: bytes) -> bool:
    """Prove byte offsets can be interpreted with the local UTF-8 tag scanner.

    Expat accepts multiple XML encodings. Its byte offsets are still useful
    in other encodings, but this scanner must not reinterpret UTF-16 bytes or
    declared legacy encodings as UTF-8. No encoding guessing is attempted.
    """
    if b'\x00' in data[:128] or data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return False
    try:
        data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return False
    declaration = data[:256].lstrip(b'\xef\xbb\xbf')
    if declaration.startswith(b'<?xml'):
        import re
        end = declaration.find(b'?>')
        if end < 0: return False
        match = re.search(rb'\bencoding\s*=\s*([\"\'])([^\"\']+)\1', declaration[:end], re.I)
        if match and match.group(2).lower().replace(b'-',b'') not in (b'utf8',):
            return False
    return True


def _start_tag_bytes(data: bytes, start: int) -> tuple[int, bool, list[dict]]:
    """Physical, quote-aware UTF-8 XML start-tag scanner. Never resolves entities.

    Expat proves well-formedness separately. A bad or ambiguous physical scan
    fails closed: no made-up byte ranges are attached to the semantic node.
    """
    if data[start:start+1] != b'<' or data[start:start+2] in (b'</', b'<!', b'<?'):
        raise XMLReaderUnavailable('xml_start_tag_bytes_not_proven')
    i=start+1;n=len(data)
    quote=None
    while i<n:
        ch=data[i]
        if quote is not None:
            if ch==quote: quote=None
        elif ch in (34,39): quote=ch
        elif ch==62:break
        i+=1
    if i>=n or quote is not None:
        raise XMLReaderUnavailable('xml_start_tag_not_closed')
    end=i+1
    inside=data[start+1:i]
    selfclosing=inside.rstrip().endswith(b'/')
    # A start tag always begins with a QName. Find its end, then scan
    # physical attribute names/values without expanding entities.
    j=start+1
    while j<i and data[j] not in b' \t\r\n/>':j+=1
    raw_qname=data[start+1:j]
    attrs=[]
    while j<i:
        while j<i and data[j] in b' \t\r\n':j+=1
        if j>=i or data[j:j+1]==b'/':break
        a_start=j
        while j<i and data[j] not in b' \t\r\n=/>':j+=1
        if j==a_start: raise XMLReaderUnavailable('xml_attribute_name_not_proven')
        rawname=data[a_start:j]
        while j<i and data[j] in b' \t\r\n':j+=1
        if j>=i or data[j]!=61:raise XMLReaderUnavailable('xml_attribute_equals_not_proven')
        j+=1
        while j<i and data[j] in b' \t\r\n':j+=1
        if j>=i or data[j] not in (34,39):raise XMLReaderUnavailable('xml_attribute_quote_not_proven')
        q=data[j];j+=1;value_start=j
        while j<i and data[j]!=q:j+=1
        if j>=i:raise XMLReaderUnavailable('xml_attribute_value_not_closed')
        value_end=j;j+=1
        attrs.append({'name':rawname.decode('utf-8'),
                      'source_byte_start':a_start,'source_byte_end':j,
                      'value_byte_start':value_start,'value_byte_end':value_end,
                      'value_is_raw_entity_escaped':True})
    qname=raw_qname.decode('utf-8')
    return end,selfclosing,attrs,qname


def _parse(data: bytes, path: str) -> dict:
    if len(data)>MAX_XML_BYTES:
        raise XMLReaderUnavailable('xml_source_size_limit')
    digest=sha256(data).hexdigest()
    file_node={'id':_file_id(path),'path':path,'name':PurePosixPath(path).name,
               'kind':'file','line':1,'confidence':'verified_file'}
    nodes=[file_node];edges=[];stack=[];serial=0
    namespace_scopes=[{'xml': XML_NAMESPACE}]
    pending_namespaces=[]
    xml_ids={}
    xml_id_issues=[]
    physical_utf8 = _physical_utf8_enabled(data)
    parser=expat.ParserCreate(namespace_separator='}')
    parser.buffer_text=False
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    def deny_doctype(*_):
        raise XMLReaderUnavailable('xml_dtd_not_allowed')

    def deny_external(*_):
        raise XMLReaderUnavailable('xml_external_entity_not_allowed')

    parser.StartDoctypeDeclHandler=deny_doctype
    parser.ExternalEntityRefHandler=deny_external

    def on_namespace(prefix, uri):
        # Expat calls this before the StartElementHandler for its own start tag.
        pending_namespaces.append({'prefix': prefix or '', 'uri': uri or ''})

    parser.StartNamespaceDeclHandler=on_namespace

    def on_open(name, attributes):
        nonlocal serial
        if len(stack)>=MAX_XML_DEPTH:
            raise XMLReaderUnavailable('xml_depth_limit')
        if len(nodes)>=MAX_XML_NODES:
            raise XMLReaderUnavailable('xml_node_limit')
        if len(attributes)>MAX_XML_ATTRIBUTES:
            raise XMLReaderUnavailable('xml_attributes_limit')
        serial+=1
        declared=pending_namespaces[:]
        pending_namespaces.clear()
        bindings=namespace_scopes[-1].copy()
        for item in declared:
            bindings[item['prefix']]=item['uri']
        namespace_scopes.append(bindings)
        line=parser.CurrentLineNumber
        # Expat byte offsets refer to the input encoding. They are not
        # Unicode codepoints or browser UTF-16 columns.
        column=parser.CurrentColumnNumber+1
        start=parser.CurrentByteIndex
        canonical=_xml_name(name)
        parent=stack[-1] if stack else file_node
        node={'id':_node_id(path,f'xml:{serial}:{start}:{canonical}',line,column),
              'path':path,'name':canonical,'kind':'xml_element','qualname':canonical,
              'line':line,'column':column,'source_byte_start':start,
              'parent':parent['id'],'confidence':'verified_xml_structure',
              'attributes':{_xml_name(k):v for k,v in attributes.items()},
              'namespace_declarations':declared,
              'attribute_source_spans':'not_proven_expat_returns_values_only'}
        parsed_attrs=node['attributes']
        if XML_ID in parsed_attrs:
            normalized=_normalized_xml_id(parsed_attrs[XML_ID])
            node['xml_id']=normalized
            if not _valid_ncname(normalized):
                node['xml_id_status']='invalid_ncname'
                xml_id_issues.append({'reason':'invalid_xml_id_ncname','value':normalized,
                                      'line':line,'node_id':node['id']})
            elif normalized in xml_ids:
                node['xml_id_status']='duplicate'
                xml_ids[normalized]['xml_id_status']='duplicate'
                xml_id_issues.append({'reason':'duplicate_xml_id','value':normalized,
                                      'line':line,'node_id':node['id'],
                                      'first_node_id':xml_ids[normalized]['id']})
            else:
                node['xml_id_status']='unique'
                xml_ids[normalized]=node
        if physical_utf8:
            # If Expat's byte position and the physical lexer disagree,
            # retain the Expat hierarchy but make no source-span claim.
            try:
                tag_end, selfclosing, physical_attrs, qname = _start_tag_bytes(data,start)
            except (XMLReaderUnavailable, UnicodeDecodeError):
                node['physical_span_status']='not_proven'
            else:
                try:
                    if _expanded_name(qname, bindings, attribute=False) != canonical:
                        raise XMLReaderUnavailable('xml_element_qname_mismatch')
                    actual_attrs=set()
                    declarations=set()
                    for item in physical_attrs:
                        source_name=item['name']
                        if source_name=='xmlns' or source_name.startswith('xmlns:'):
                            prefix=source_name[6:] if source_name!='xmlns' else ''
                            if prefix not in bindings and prefix!='':
                                raise XMLReaderUnavailable('xml_namespace_declaration_mismatch')
                            item['namespace_declaration_prefix']=prefix
                            declarations.add(prefix)
                        else:
                            expanded=_expanded_name(source_name, bindings, attribute=True)
                            if expanded not in parsed_attrs:
                                raise XMLReaderUnavailable('xml_attribute_name_mismatch')
                            item['expanded_name']=expanded
                            item['normalized_value']=parsed_attrs[expanded]
                            actual_attrs.add(expanded)
                    if actual_attrs!=set(parsed_attrs) or declarations!={d['prefix'] for d in declared}:
                        raise XMLReaderUnavailable('xml_attribute_or_namespace_count_mismatch')
                except XMLReaderUnavailable:
                    node['physical_span_status']='not_proven'
                else:
                    node.update(physical_span_status='verified_utf8_byte_spans',
                                source_qname=qname,
                                start_tag_byte_start=start, start_tag_byte_end=tag_end,
                                self_closing=selfclosing,
                                physical_attributes=physical_attrs,
                                attribute_source_spans='verified_utf8_byte_spans')
        else:
            node['physical_span_status']='not_available_for_encoding'
        nodes.append(node)
        edges.append({'source':parent['id'],'target':node['id'],
                      'source_path':path,'target_path':path,
                      'relation':'contains','resolution':'verified_xml_structure',
                      'analyzer':'xml-expat-structure',
                      'evidence':{'path':path,'line':line,'column':column,
                                  'source_sha256':digest,'utf8_byte_offset':start}})
        stack.append(node)

    def on_close(_name):
        if not stack:
            raise XMLReaderUnavailable('xml_stack_underflow')
        node=stack.pop()
        namespace_scopes.pop()
        node['end_line']=parser.CurrentLineNumber
        node['end_column']=parser.CurrentColumnNumber+1
        # CurrentByteIndex at a non-self-closing end tag points to its '<'.
        # For a self closing tag it points after '/>'. No presumed exact end
        # span is attached until both forms have a proven physical lexer.
        node['end_byte_index_expat']=parser.CurrentByteIndex
        if node.get('physical_span_status')=='verified_utf8_byte_spans':
            if node['self_closing']:
                node['source_byte_end']=node['start_tag_byte_end']
            else:
                ending=parser.CurrentByteIndex
                # XML end tags contain no attributes or quotes.
                if data[ending:ending+2]==b'</':
                    e=data.find(b'>',ending+2)
                    if e>=0:
                        node['end_tag_byte_start']=ending
                        node['end_tag_byte_end']=e+1
                        node['source_byte_end']=e+1
                    else:
                        node['physical_span_status']='end_tag_not_proven'
                else:
                    node['physical_span_status']='end_tag_not_proven'

    parser.StartElementHandler=on_open
    parser.EndElementHandler=on_close
    try:
        parser.Parse(data,True)
    except expat.ExpatError as exc:
        raise XMLReaderUnavailable(f'xml_not_well_formed:{exc.code}:{exc.lineno}:{exc.offset}') from exc
    if stack:
        raise XMLReaderUnavailable('xml_unclosed_element')
    return {'nodes':nodes,'edges':edges,'hash':digest,
            'xml_id_index':{value:node['id'] for value,node in xml_ids.items()
                            if node['xml_id_status']=='unique'},
            'xml_id_issues':xml_id_issues,
            'parser':'xml_expat_source_structure','source_coordinates':'expat_input_encoded_byte_offsets',
            'caution':'XML 1.0 namespace-aware element structure and xml:id diagnostics; no DTD, external entity, schema, IDREF or XInclude resolution.',
            'declaration_limitations':['no_DTD','no_external_entity_resolution','no_schema_validation',
                                       'physical_attribute_spans_utf8_only','no_XInclude_execution','not_full_semantic_graph']}


def analyze_xml(root: Path,candidates: dict,_graph: dict) -> tuple[dict,list[dict]]:
    files={};warnings=[]
    for path,entry in sorted(candidates.items()):
        if entry.get('language')!='xml':continue
        try:
            file=_allowed(root,path)
            if file.stat().st_size>MAX_XML_BYTES:
                raise XMLReaderUnavailable('xml_source_size_limit')
            data=file.read_bytes()
            files[path]=_parse(data,path)
        except (XMLReaderUnavailable, OSError, ValueError) as exc:
            warnings.append({'path':path,'reason':'xml_source_rejected',
                             'details':str(exc)[:180],
                             'semantic_nodes_published':False})
    return files,warnings
