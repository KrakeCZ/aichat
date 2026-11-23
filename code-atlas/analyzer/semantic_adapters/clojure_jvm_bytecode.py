"""Read-only JVM classfile evidence for Clojure interop syntax.

This module NEVER loads classes, runs javac, Java, the target project's code, or
any build system. It interprets bounded public member *declarations* from
classes already present in an explicitly supplied, checksum-validated classpath.
Java overload resolution, macro expansion, dynamic dispatch, type inference,
and execution remain unproved. Classfile arities and declared supertypes are
read from bytes, never treated as compiler-selected invocation targets.
"""
from __future__ import annotations

import hashlib
import re
import struct
import zipfile
from pathlib import Path

MAX_CLASS_BYTES = 2_000_000
MAX_JAR_BYTES = 200_000_000
MAX_JAR_MANIFEST_BYTES = 50_000
MAX_SUPPORTED_TARGET_JAVA = 27

MAX_INTEROP_CLASSES = 96
MAX_INTEROP_SITES = 800
MAX_HIERARCHY_DEPTH = 16
MAX_HIERARCHY_CLASSES = 256
MAX_INTERFACE_DEPTH = 16
MAX_MEMBERS = 4096
MAX_CONSTANT_POOL = 20000
_CLASS_NAME = re.compile(r'[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\Z', re.ASCII)


class BytecodeUnavailable(ValueError):
    pass


class _Bytes:
    def __init__(self, data: bytes):
        if not 10 <= len(data) <= MAX_CLASS_BYTES:
            raise BytecodeUnavailable('class_file_size_invalid')
        self.data = data
        self.pos = 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise BytecodeUnavailable('truncated_class_file')
        result = self.data[self.pos:self.pos+n]
        self.pos += n
        return result

    def u1(self):
        return self.take(1)[0]

    def u2(self):
        return struct.unpack('>H', self.take(2))[0]

    def u4(self):
        return struct.unpack('>I', self.take(4))[0]


def _modified_utf8(raw: bytes) -> str:
    """Decode JVMS 4.4.7 modified UTF-8, never ordinary UTF-8.

    A classfile encodes UTF-16 code units: U+0000 is C0 80, supplementary
    characters are pairs of three-byte surrogate units, and 4-byte UTF-8 is
    forbidden. Reject invalid/truncated/overlong encodings without guessing.
    """
    units = bytearray()
    i = 0
    while i < len(raw):
        lead = raw[i]
        if 1 <= lead <= 0x7F:
            unit, advance = lead, 1
        elif 0xC0 <= lead <= 0xDF:
            if i + 1 >= len(raw) or raw[i+1] & 0xC0 != 0x80:
                raise BytecodeUnavailable('invalid_modified_utf8')
            unit = ((lead & 0x1F) << 6) | (raw[i+1] & 0x3F)
            if unit < 0x80 and (lead, raw[i+1]) != (0xC0, 0x80):
                raise BytecodeUnavailable('invalid_modified_utf8')
            advance = 2
        elif 0xE0 <= lead <= 0xEF:
            if (i + 2 >= len(raw) or raw[i+1] & 0xC0 != 0x80
                    or raw[i+2] & 0xC0 != 0x80):
                raise BytecodeUnavailable('invalid_modified_utf8')
            unit = ((lead & 0x0F) << 12) | ((raw[i+1] & 0x3F) << 6) | (raw[i+2] & 0x3F)
            if unit < 0x800:
                raise BytecodeUnavailable('invalid_modified_utf8')
            advance = 3
        else:
            raise BytecodeUnavailable('invalid_modified_utf8')
        units.extend(unit.to_bytes(2, 'big'))
        i += advance
    # surrogatepass preserves literal isolated UTF-16 units if they appear in
    # arbitrary CONSTANT_Utf8 string data. Names/descriptors are validated
    # separately before they are used as class/method identities.
    return units.decode('utf-16-be', errors='surrogatepass')


def _method_parameter_descriptors(descriptor: str) -> tuple[str, ...]:
    """JVMS 4.3.3 method descriptor grammar with 255-slot bound.

    Returns raw parameter descriptors, including array dimensions. Each long
    or double occupies two JVM parameter slots, even though it is one Clojure
    call argument. No return/void or malformed descriptor is accepted.
    """
    if not isinstance(descriptor, str) or not descriptor.startswith('('):
        raise BytecodeUnavailable('invalid_method_descriptor')
    n = len(descriptor)
    pos = 1
    parameters = []
    slots = 0

    def read_type(index, *, allow_void=False):
        start = index
        dimensions = 0
        while index < n and descriptor[index] == '[':
            index += 1
            dimensions += 1
            if dimensions > 255:
                raise BytecodeUnavailable('invalid_method_descriptor')
        if index >= n:
            raise BytecodeUnavailable('invalid_method_descriptor')
        ch = descriptor[index]
        if ch == 'V' and allow_void and dimensions == 0:
            return index+1
        if ch in 'BCDFIJSZ':
            return index+1
        if ch == 'L':
            end = descriptor.find(';', index+1)
            if end <= index+1 or any(c in descriptor[index+1:end] for c in '.;[()'):
                raise BytecodeUnavailable('invalid_method_descriptor')
            # A JVM internal binary class name uses slash separators, never
            # empty path components or an initial/final slash.
            internal = descriptor[index+1:end]
            if internal.startswith('/') or internal.endswith('/') or '//' in internal:
                raise BytecodeUnavailable('invalid_method_descriptor')
            return end+1
        raise BytecodeUnavailable('invalid_method_descriptor')

    while pos < n and descriptor[pos] != ')':
        start = pos
        pos = read_type(pos)
        part = descriptor[start:pos]
        slots += 2 if part in ('J', 'D') else 1
        if slots > 255:
            raise BytecodeUnavailable('method_descriptor_too_many_parameter_slots')
        parameters.append(part)
    if pos >= n or descriptor[pos] != ')':
        raise BytecodeUnavailable('invalid_method_descriptor')
    pos = read_type(pos+1, allow_void=True)
    if pos != n:
        raise BytecodeUnavailable('invalid_method_descriptor')
    return tuple(parameters)


def _descriptor_arity(descriptor):
    return len(_method_parameter_descriptors(descriptor))


def _arity_compatibility(method: dict, supplied_count: int) -> str | None:
    """Distinguish exact descriptor arity from a possible varargs expansion.

    ACC_VARARGS alone does not prove that a particular Clojure call selected
    Java varargs or which overload won. The last parameter must be an array.
    """
    arguments = _method_parameter_descriptors(method['descriptor'])
    if supplied_count == len(arguments):
        return 'declared_parameter_count_match'
    if (method.get('varargs') and arguments and arguments[-1].startswith('[')
            and supplied_count >= len(arguments)-1):
        return 'varargs_expansion_candidate'
    return None


def parse_class(data: bytes, expected_internal_name: str) -> dict:
    """Return public declarations ONLY; never interpret bytecode instructions."""
    reader = _Bytes(data)
    if reader.u4() != 0xCAFEBABE:
        raise BytecodeUnavailable('bad_class_magic')
    minor, major = reader.u2(), reader.u2()
    if not 45 <= major <= 80:
        raise BytecodeUnavailable('unsupported_class_version')
    count = reader.u2()
    if not 2 <= count <= MAX_CONSTANT_POOL:
        raise BytecodeUnavailable('constant_pool_limit')
    pool = [None]*count
    i = 1
    while i < count:
        tag = reader.u1()
        if tag == 1:
            size = reader.u2()
            raw = reader.take(size)
            pool[i] = ('utf8', _modified_utf8(raw))
        elif tag in (3, 4):
            reader.take(4)
        elif tag in (5, 6):
            reader.take(8)
            i += 1
        elif tag in (7, 8, 16, 19, 20):
            pool[i] = ('class' if tag == 7 else 'other', reader.u2())
        elif tag in (9, 10, 11, 12, 17, 18):
            reader.take(4)
        elif tag == 15:
            reader.take(3)
        else:
            raise BytecodeUnavailable('unknown_constant_pool_tag')
        i += 1

    def utf8(idx):
        if not 0 < idx < len(pool) or not pool[idx] or pool[idx][0] != 'utf8':
            raise BytecodeUnavailable('invalid_utf8_reference')
        return pool[idx][1]

    flags, this_class, parent_ref = reader.u2(), reader.u2(), reader.u2()
    if not 0 < this_class < len(pool) or not pool[this_class] or pool[this_class][0] != 'class':
        raise BytecodeUnavailable('invalid_class_reference')
    internal = utf8(pool[this_class][1])
    if internal != expected_internal_name:
        raise BytecodeUnavailable('class_identity_mismatch')
    def class_ref(ref):
        if not 0 < ref < len(pool) or not pool[ref] or pool[ref][0] != 'class':
            raise BytecodeUnavailable('invalid_supertype_reference')
        internal_name = utf8(pool[ref][1])
        if not internal_name or '.' in internal_name or internal_name.startswith('/') or '..' in internal_name:
            raise BytecodeUnavailable('invalid_supertype_internal_name')
        return internal_name.replace('/', '.')

    superclass = class_ref(parent_ref) if parent_ref else None
    if superclass is None and internal != 'java/lang/Object':
        raise BytecodeUnavailable('invalid_null_superclass')
    n_interfaces=reader.u2()
    if n_interfaces > 1024:
        raise BytecodeUnavailable('class_interface_limit')
    interfaces = [class_ref(reader.u2()) for _ in range(n_interfaces)]

    def members():
        total = reader.u2()
        if total > MAX_MEMBERS:
            raise BytecodeUnavailable('class_member_limit')
        found = []
        for _ in range(total):
            access, name_ref, desc_ref = reader.u2(), reader.u2(), reader.u2()
            name, desc = utf8(name_ref), utf8(desc_ref)
            attributes = reader.u2()
            if attributes > 1024:
                raise BytecodeUnavailable('class_attribute_limit')
            for _ in range(attributes):
                reader.u2()
                size = reader.u4()
                reader.take(size)
            if access & 0x0001:  # only public declarations
                found.append({'name': name, 'descriptor': desc,
                              'static': bool(access & 0x0008),
                              'varargs': bool(access & 0x0080),
                              'synthetic': bool(access & 0x1000)})
        return found

    fields = members()
    methods = members()
    # Do not assert attributes are complete, but validate their envelope.
    class_attrs = reader.u2()
    for _ in range(class_attrs):
        reader.u2()
        reader.take(reader.u4())
    if reader.pos != len(data):
        raise BytecodeUnavailable('class_trailing_bytes')
    return {'class': internal.replace('/', '.'), 'major_version': major, 'minor_version': minor,
            'superclass': superclass, 'interfaces': interfaces,
            'public_class': bool(flags & 0x0001), 'interface': bool(flags & 0x0200),
            'sha256': hashlib.sha256(data).hexdigest(),
            'fields': fields, 'methods': methods}


def _mr_jar_manifest(archive: zipfile.ZipFile) -> bool:
    """Read the *main* JAR manifest section with continuation lines.

    Per-entry sections do not activate Multi-Release. Conflicting or duplicate
    relevant headers are not stable enough to prove a selected Java class.
    """
    matching = [item for item in archive.infolist()
                if item.filename.upper() == 'META-INF/MANIFEST.MF']
    if not matching:
        return False
    if len(matching) != 1 or matching[0].is_dir():
        raise BytecodeUnavailable('jar_manifest_ambiguous')
    member = matching[0]
    if member.file_size > MAX_JAR_MANIFEST_BYTES:
        raise BytecodeUnavailable('jar_manifest_exceeds_limit')
    raw = archive.read(member)
    if len(raw) > MAX_JAR_MANIFEST_BYTES:
        raise BytecodeUnavailable('jar_manifest_exceeds_limit')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        raise BytecodeUnavailable('jar_manifest_invalid_utf8') from None
    main = re.split(r'\r\n|\r|\n', text)
    headers = []
    for line in main:
        if not line:
            break  # named entry sections are NOT main attributes
        if line.startswith(' '):
            if not headers:
                raise BytecodeUnavailable('jar_manifest_invalid_continuation')
            key, val = headers[-1]
            headers[-1] = (key, val + line[1:])
        elif ': ' in line:
            key, value = line.split(': ', 1)
            headers.append((key.lower(), value))
        else:
            raise BytecodeUnavailable('jar_manifest_invalid_header')
    values = [value for key, value in headers if key == 'multi-release']
    if len(values) > 1:
        raise BytecodeUnavailable('jar_manifest_duplicate_multi_release')
    return len(values) == 1 and values[0].lower() == 'true'


def _verified_jar(p: Path, expected: str, *, cache: dict | None) -> bool:
    """Check the complete immutable-entry checksum immediately before use.

    The snapshot's original verification is insufficient if a JAR was replaced
    after it was loaded. A per-analysis cache uses device/inode/size/mtime/ctime
    to avoid rehashing the same large JAR for every class in a hierarchy.
    This detects normal changes; it is not an atomic filesystem snapshot and
    cannot defeat a concurrent, adversarial in-place rewrite.
    """
    stat = p.stat()
    if stat.st_size > MAX_JAR_BYTES or not p.is_file():
        raise BytecodeUnavailable('classpath_jar_exceeds_limit')
    stamp = (stat.st_dev, stat.st_ino, stat.st_size,
             stat.st_mtime_ns, stat.st_ctime_ns)
    key = str(p)
    if cache is not None and cache.get(key) == (stamp, expected):
        return True
    digest = hashlib.sha256()
    with p.open('rb') as handle:
        while True:
            chunk = handle.read(1024*1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != expected:
        return False
    if p.stat().st_size != stat.st_size or p.stat().st_mtime_ns != stat.st_mtime_ns:
        return False
    if cache is not None:
        cache[key] = (stamp, expected)
    return True


def _versioned_class_resource(archive: zipfile.ZipFile, target: str,
                              target_jvm_major: int | None, is_multi: bool):
    """JVMS/JAR spec: highest eligible N >= 9, falling back to root.

    Return (resource, selected N). The caller checks duplicate members and
    the contents; no ``ZipFile.getinfo`` duplicate ambiguity is concealed.
    """
    names = {i.filename for i in archive.infolist()}
    base_exists = target in names
    if not is_multi:
        return (target if base_exists else None, None)
    versioned = []
    suffix = '/' + target
    prefix = 'META-INF/versions/'
    for name in names:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        release = name[len(prefix):-len(suffix)]
        # The supported target JVM is <= 27. A huge numeric ZIP path
        # component must never reach int() (Python's digit guard is another
        # potential source of an uncaught ValueError).
        if (1 <= len(release) <= 3 and release.isascii() and release.isdecimal() and
                release[0] != '0' and int(release) >= 9):
            versioned.append((int(release), name))
    if target_jvm_major is None:
        if base_exists or versioned:
            raise BytecodeUnavailable('multi_release_jar_requires_target_jvm_version')
        return None, None
    eligible = [x for x in versioned if x[0] <= target_jvm_major]
    if eligible:
        version, resource = max(eligible)
        return resource, version
    return (target if base_exists else None, None)


def class_from_classpath(snapshot: dict, binary_name: str, *,
                         target_jvm_major: int | None = None,
                         verified_jars: dict | None = None) -> dict:
    """Inspect FIRST classpath provider under explicit, claimed Java target.

    Target Java is a user assertion, not a detected runtime or an assertion
    that a compiler used the supplied classpath. A MR JAR is deliberately
    unavailable without it.
    """
    if not _CLASS_NAME.fullmatch(binary_name):
        return {'status': 'invalid_class_name'}
    if (target_jvm_major is not None and
            (type(target_jvm_major) is not int or not 8 <= target_jvm_major <= MAX_SUPPORTED_TARGET_JAVA)):
        return {'status': 'invalid_target_jvm_major'}
    target = binary_name.replace('.', '/') + '.class'
    if not snapshot or snapshot.get('status') != 'supplied_snapshot_validated_against_local_files':
        return {'status': 'classpath_not_supplied'}
    for entry in snapshot['entries']:
        p = Path(entry['path'])
        try:
            if p.is_symlink() or p.resolve() != p:
                return {'status': 'classpath_entry_changed'}
            selected_resource, selected_release, mr = target, None, False
            if entry['kind'] == 'jar':
                if not _verified_jar(p, entry['sha256'], cache=verified_jars):
                    return {'status': 'classpath_entry_hash_changed_since_snapshot'}
                with zipfile.ZipFile(p) as archive:
                    mr = _mr_jar_manifest(archive)
                    selected_resource, selected_release = _versioned_class_resource(
                        archive, target, target_jvm_major, mr)
                    root_peer_exists = target in {item.filename for item in archive.infolist()}
                    if selected_resource is None:
                        continue
                    matching = [item for item in archive.infolist()
                                if item.filename == selected_resource]
                    if len(matching) != 1:
                        return {'status': 'classfile_duplicate_jar_resource_ambiguous'}
                    member = matching[0]
                    if member.file_size > MAX_CLASS_BYTES or member.is_dir():
                        return {'status': 'class_file_exceeds_limit'}
                    with archive.open(member) as handle:
                        data = handle.read(MAX_CLASS_BYTES+1)
            else:
                root_peer_exists = False
                file_path = p / target
                current=p
                for component in Path(target).parts:
                    current = current / component
                    if current.is_symlink():
                        return {'status': 'class_file_path_symlink'}
                if not file_path.is_file():
                    continue
                if not file_path.resolve().is_relative_to(p.resolve()):
                    return {'status': 'class_file_path_escape'}
                if file_path.stat().st_size > MAX_CLASS_BYTES:
                    return {'status': 'class_file_exceeds_limit'}
                data = file_path.read_bytes()
            result = parse_class(data, binary_name.replace('.', '/'))
            if target_jvm_major is not None and result['major_version'] > target_jvm_major + 44:
                return {'status': 'classfile_exceeds_explicit_target_jvm'}
            if selected_release is not None and result['major_version'] > selected_release + 44:
                return {'status': 'multi_release_classfile_exceeds_version_directory'}
            if selected_release is not None and result['public_class'] and not root_peer_exists:
                return {'status': 'multi_release_public_class_missing_root_peer'}
            if target_jvm_major is not None and result['minor_version'] == 65535:
                return {'status': 'preview_classfile_requires_unverified_runtime_flags'}
            return {'status': 'classfile_public_members_read',
                    'classpath_position': entry['position'],
                    'classpath_kind': entry['kind'],
                    'classpath_entry_basename': p.name,
                    'classpath_entry_sha256': entry.get('sha256'),
                    'selected_class_resource': selected_resource,
                    'multi_release_jar': mr,
                    'selected_multi_release_version': selected_release,
                    'target_jvm_major_user_asserted': target_jvm_major,
                    **result}
        except BytecodeUnavailable as e:
            # Deliberate provenance failures must not permit a later provider.
            reason = str(e)[:100]
            if reason == 'multi_release_jar_requires_target_jvm_version':
                return {'status': reason}
            return {'status': 'classfile_unavailable', 'reason': reason}
        except (OSError, RuntimeError, zipfile.BadZipFile, EOFError):
            return {'status': 'classfile_unavailable', 'reason':'classpath_resource_io_failure'}
    return {'status': 'classfile_not_in_explicit_classpath'}


def java_imports(forms) -> tuple[dict[str, str], list[str]]:
    """Recognize only literal JVM :import shapes, preserving conflicts."""
    from .clojure_forms import _top, _sym
    ns_forms = [form for form in forms if _top(form, 'ns')]
    if len(ns_forms) != 1:
        return {}, []
    result: dict[str,str] = {}
    ambiguous: set[str] = set()
    for clause in ns_forms[0].children[2:]:
        if not _top(clause, ':import'):
            continue
        for part in clause.children[1:]:
            names=[]
            if part.kind == 'atom' and _CLASS_NAME.fullmatch(_sym(part)) and '.' in _sym(part):
                names.append(_sym(part))
            elif part.kind in ('list','vector') and len(part.children) > 1:
                package = _sym(part.children[0])
                if not package or not _CLASS_NAME.fullmatch(package):
                    continue
                for child in part.children[1:]:
                    cls = _sym(child)
                    if cls and _CLASS_NAME.fullmatch(cls) and '.' not in cls:
                        names.append(package+'.'+cls)
            for fullname in names:
                simple = fullname.rsplit('.',1)[-1]
                if simple in ambiguous:
                    continue
                if simple in result and result[simple] != fullname:
                    del result[simple]
                    ambiguous.add(simple)
                else:
                    result[simple] = fullname
    return result, sorted(ambiguous)


# Clojure 1.12 qualified method values and ^[tag ...] / :param-tags.
# This is literal syntax-only signature filtering against JVM classfiles,
# never a claim of compiler overload selection or executable code.
_PRIMITIVE_TAGS = {
    'boolean':'Z', 'byte':'B', 'char':'C', 'short':'S', 'int':'I',
    'long':'J', 'float':'F', 'double':'D',
    'booleans':'[Z', 'bytes':'[B', 'chars':'[C', 'shorts':'[S',
    'ints':'[I', 'longs':'[J', 'floats':'[F', 'doubles':'[D',
    'objects':'[Ljava/lang/Object;',
}
_JAVA_LANG_TAGS = frozenset({'String','Object','Number','Boolean','Byte',
    'Character','Short','Integer','Long','Float','Double','Class','Throwable',
    'Exception','RuntimeException','Iterable','Comparable','StringBuilder'})


def _literal_param_tags(atom, imports):
    """Return (tag descriptors, issue), where None means no annotation.

    A wildcard is kept as ``_``. Reject unresolved tags rather than pretending
    that a guessed Java class/array is a compiler-selected signature.
    """
    explicit = []
    for metadata in atom.metadata:
        raw = None
        if metadata.kind == 'vector':
            raw = metadata
        elif metadata.kind == 'map':
            pairs = metadata.children
            keys = [pairs[i].value for i in range(0,len(pairs)-1,2)
                    if pairs[i].kind == 'atom']
            if keys.count(':param-tags') > 1:
                return None, 'ambiguous_repeated_param_tags'
            for i in range(0,len(pairs)-1,2):
                if pairs[i].kind == 'atom' and pairs[i].value == ':param-tags':
                    raw = pairs[i+1]
        if raw is not None:
            explicit.append(raw)
    if not explicit:
        return None, None
    if len(explicit) != 1 or explicit[0].kind != 'vector':
        return None, 'ambiguous_or_nonliteral_param_tags'
    if len(explicit[0].children) > 255:
        return None, 'param_tags_over_limit'
    tags=[]
    for form in explicit[0].children:
        if form.kind != 'atom':
            return None, 'unsupported_param_tag_shape'
        tag = form.value
        if tag == '_':
            tags.append('_')
        elif tag in _PRIMITIVE_TAGS:
            tags.append(_PRIMITIVE_TAGS[tag])
        elif tag in imports:
            tags.append('L'+imports[tag].replace('.','/')+';')
        elif tag in _JAVA_LANG_TAGS:
            tags.append('Ljava/lang/'+tag+';')
        elif _CLASS_NAME.fullmatch(tag) and '.' in tag:
            tags.append('L'+tag.replace('.','/')+';')
        else:
            return None, 'unresolved_or_unsupported_param_tag'
    return tuple(tags), None


def _param_tags_match(method, tags):
    if tags is None:
        return True
    params = _method_parameter_descriptors(method['descriptor'])
    return len(params) == len(tags) and all(
        tag == '_' or tag == desc for tag,desc in zip(tags,params))


def member_sites(forms, imports: dict, aliases: dict, alias_guards: dict,
                 referred: dict, shadowed_core: frozenset,
                 core_macro_aliases: dict) -> list[dict]:
    """Walk known lexical scopes. A call-head site is still only a candidate."""
    from .clojure_forms import Form, _definition
    from .clojure_scope import referred_call_sites
    sites=[]
    for top in forms:
        info = _definition(top)
        if not info or info[1] not in ('function','macro','variable'):
            continue
        owner,kind,_part=info
        body=list(top.children[2:])
        if kind in ('function','macro'):
            if body and body[0].kind=='string':body.pop(0)
            if body and body[0].kind=='map':body.pop(0)
            if body and body[-1].kind=='map' and body[0].kind=='list':body.pop()
            marker=Form('internal_function_scope','fn*',(),top.line,top.col)
            body=[Form('list','(',tuple([marker,*body]),top.line,top.col)]
        call_sites,_=referred_call_sites(tuple(body), frozenset(), None,
            include_qualified=True, core_shadowed=shadowed_core,
            macro_aliases=core_macro_aliases)
        # Original source forms carry exact site coordinates. Only heads
        # discovered by the scoped scanner are eligible for inspection.
        by_head={}
        stack=list(body)
        while stack:
            current=stack.pop()
            if current.kind=='list' and current.children and current.children[0].kind=='atom':
                head=current.children[0]
                by_head[(head.line,head.col)] = current
            stack.extend(current.children)
        for hit in call_sites:
            head=hit.head
            if '/' not in head.value:
                continue
            prefix,name=head.value.split('/',1)
            if prefix in aliases or prefix in alias_guards or not name:
                continue
            fullname=imports.get(prefix)
            if fullname is None and '.' in prefix and _CLASS_NAME.fullmatch(prefix):
                fullname=prefix
            if not fullname or (prefix not in imports and not fullname.split('.')[-1][:1].isupper()):
                continue
            form=by_head.get((head.line,head.col))
            if form is None:
                continue
            # JVM 1.12 Class/.instanceMember is not a static invocation.
            instance=name.startswith('.') and len(form.children)>=2
            if name.startswith('.'):
                name=name[1:]
            if not name or not re.fullmatch(r'[A-Za-z_$][\w$]*',name,re.ASCII):
                continue
            if len(sites)>=MAX_INTEROP_SITES:
                return sites
            # Clojure 1.12 Class/new is a constructor (not a static method).
            ctor = name == 'new' and not instance
            param_tags,param_issue=_literal_param_tags(head,imports)
            sites.append({'owner':owner,'class':fullname,
                          'member':'<init>' if ctor else name,
                          'member_kind':'constructor' if ctor else 'method',
                          'static':False if ctor else not instance,
                          'argc':len(form.children)-(2 if instance else 1),
                          'reference_kind':'call_head',
                          'param_tag_descriptors':param_tags,
                          'param_tags_unresolved_reason':param_issue,
                          'line':head.line,'column':head.col,
                          'end_line':head.end_line,'end_column':head.end_col,
                          'phase':'macro_expansion' if kind=='macro' else 'runtime_candidate'})
        value_sites,_=referred_call_sites(tuple(body), frozenset(), None,
            include_qualified=True, include_values=True, include_call_heads=False,
            core_shadowed=shadowed_core, macro_aliases=core_macro_aliases)
        for hit in value_sites:
            name=hit.head.value
            if '/' not in name:
                continue
            prefix,member=name.split('/',1)
            if (prefix in aliases or prefix in alias_guards or
                    not re.fullmatch(r'\.?[A-Za-z_$][\w$]*',member,re.ASCII)):
                continue
            fullname=imports.get(prefix)
            if fullname is None and '.' in prefix and _CLASS_NAME.fullmatch(prefix):
                fullname=prefix
            if not fullname or (prefix not in imports and not fullname.split('.')[-1][:1].isupper()):
                continue
            if len(sites)>=MAX_INTEROP_SITES:
                return sites
            param_tags,param_issue=_literal_param_tags(hit.head,imports)
            # Class/method may denote a static field or, since Clojure 1.12,
            # a qualified method value. Never silently prefer either meaning.
            # Method values must NOT pretend the number of call args is zero.
            ctor=member=='new'
            qualified_instance=member.startswith('.') and len(member)>1
            if ctor or qualified_instance or param_tags is not None or param_issue:
                alternatives=(('constructor','<init>',False) if ctor else
                              ('method',member[1:] if qualified_instance else member,
                               not qualified_instance),)
            else:
                # Preserve a SINGLE site for the ambiguous static symbol.
                # Classfile inspection below can prove a method declaration,
                # but the compiler's field-vs-method choice is never asserted.
                alternatives=(('field',member,True),)
            for member_kind,member_name,is_static in alternatives:
                if len(sites)>=MAX_INTEROP_SITES:
                    return sites
                sites.append({'owner':owner,'class':fullname,'member':member_name,
                              'member_kind':member_kind,'static':is_static,'argc':None,
                              'reference_kind':'qualified_member_value',
                              'param_tag_descriptors':param_tags,
                              'param_tags_unresolved_reason':param_issue,
                              'qualified_static_ambiguous':not (ctor or qualified_instance or
                                  param_tags is not None or param_issue),
                              'line':hit.head.line,'column':hit.head.col,
                              'end_line':hit.head.end_line,'end_column':hit.head.end_col,
                              'phase':'macro_expansion' if kind=='macro' else 'runtime_candidate'})
    return sites


def _class_symbol(name, imports):
    if name in imports:
        return imports[name]
    if '.' in name and _CLASS_NAME.fullmatch(name) and name.rsplit('.',1)[-1][:1].isupper():
        return name
    return None


def _tagged_binding(pattern, imports):
    """Only an explicit ^Type symbol tag: this is syntax, not type inference."""
    if pattern.kind != 'atom':
        return None
    for metadata in pattern.metadata:
        if metadata.kind == 'atom':
            result = _class_symbol(metadata.value, imports)
            if result:
                return result
    return None


def literal_typed_interop_sites(forms, imports):
    """Scoped explicit Java hints and constructor syntax, no compiler execution.

    Handles fn parameters and sequential let/loop bindings with literal tags.
    Unknown destructuring, user-defined binding macros, and reader-dispatch
    templates are not given any inferred type or guaranteed call target.
    """
    from .clojure_forms import _definition
    from .clojure_scope import _fn_arities, binding_names
    out=[]
    limit=MAX_INTEROP_SITES

    def add(owner, class_name, member, member_kind, static, argc, form, phase, *, receiver_hint=None):
        if len(out)>=limit:
            return
        out.append({'owner':owner,'class':class_name,'member':member,
                    'member_kind':member_kind,'static':static,'argc':argc,
                    'line':form.line,'column':form.col,
                    'end_line':form.end_line,'end_column':form.end_col,
                    'phase':phase,
                    'receiver_type_hint':receiver_hint,
                    'type_evidence':('explicit_lexical_type_tag_not_compiler_inferred'
                                     if receiver_hint else 'literal_class_constructor_syntax')})

    def walk(form, env, owner, phase, depth=0):
        if depth>70 or len(out)>=limit:
            return
        if form.kind=='opaque':
            return
        if form.kind != 'list' or not form.children:
            for child in form.children:walk(child,env,owner,phase,depth+1)
            return
        head=form.children[0]
        h=head.value if head.kind=='atom' else ''
        children=form.children
        if h in {'quote','clojure.core/quote','comment','clojure.core/comment'}:
            return
        if h in {'let','let*','loop','loop*'} and len(children)>=3 and children[1].kind=='vector':
            bindings=children[1].children
            if len(bindings)%2:
                return
            local=dict(env)
            for i in range(0,len(bindings),2):
                pattern,value=bindings[i:i+2]
                names=binding_names(pattern)
                if names is None:
                    return
                walk(value,local,owner,phase,depth+1)
                for name in names:local[name]=None
                if pattern.kind=='atom':
                    local[pattern.value]=_tagged_binding(pattern,imports)
            for expression in children[2:]:walk(expression,local,owner,phase,depth+1)
            return
        if h in {'fn','fn*'}:
            rest=list(children[1:])
            local=dict(env)
            if rest and rest[0].kind=='atom':
                local[rest[0].value]=None
                rest=rest[1:]
            arities=_fn_arities(tuple(rest))
            if arities is None:
                return
            for params,expressions in arities:
                bindings=dict(local)
                for param in params.children:
                    if param.kind=='atom' and param.value=='&':
                        continue
                    names=binding_names(param)
                    if names is None:return
                    for name in names:bindings[name]=None
                    if param.kind=='atom':
                        bindings[param.value]=_tagged_binding(param,imports)
                for expression in expressions:walk(expression,bindings,owner,phase,depth+1)
            return
        if h=='letfn' and len(children)>=3 and children[1].kind=='vector':
            definitions=children[1].children
            if not definitions or any(item.kind!='list' or len(item.children)<3 or
                                      item.children[0].kind!='atom' for item in definitions):
                return
            names=[item.children[0].value for item in definitions]
            if len(set(names))!=len(names):
                return
            scope=dict(env)
            for name in names:scope[name]=None
            for function in definitions:
                arities=_fn_arities(tuple(function.children[1:]))
                if arities is None:
                    return
                for params,expressions in arities:
                    local=dict(scope)
                    for pattern in params.children:
                        if pattern.kind=='atom' and pattern.value=='&':continue
                        bound=binding_names(pattern)
                        if bound is None:return
                        for name in bound:local[name]=None
                        if pattern.kind=='atom':
                            local[pattern.value]=_tagged_binding(pattern,imports)
                    for expression in expressions:
                        walk(expression,local,owner,phase,depth+1)
            for expression in children[2:]:
                walk(expression,scope,owner,phase,depth+1)
            return
        if h=='catch':
            if len(children)<4 or children[2].kind!='atom':
                return
            scope=dict(env)
            scope[children[2].value]=None
            for expression in children[3:]:
                walk(expression,scope,owner,phase,depth+1)
            return
        # Compiler macro expansions may bind identifiers with arbitrary
        # semantics. Do not reuse outer type hints through opaque macros.
        from .clojure_scope import OPAQUE_BINDING_MACROS
        if h in OPAQUE_BINDING_MACROS:
            return
        # Modern Classname/.instanceMethod and instance shorthand. Receiver
        # MUST be an unambiguous explicitly tagged local to infer its class.
        if h.startswith('.-') and len(children)==2:
            recv=children[1]
            if recv.kind=='atom' and (cls:=env.get(recv.value)):
                add(owner,cls,h[2:],'field',False,None,head,phase,receiver_hint=recv.value)
        elif h.startswith('.') and h not in {'.','..'} and len(children)>=2:
            recv=children[1]
            if recv.kind=='atom' and (cls:=env.get(recv.value)):
                add(owner,cls,h[1:],'method',False,len(children)-2,head,phase,
                    receiver_hint=recv.value)
        elif h=='.' and len(children)>=3:
            recv,member=children[1],children[2]
            if recv.kind=='atom' and member.kind=='atom' and (cls:=env.get(recv.value)):
                if member.value.startswith('-'):
                    add(owner,cls,member.value[1:],'field',False,None,member,phase,
                        receiver_hint=recv.value)
                elif re.fullmatch(r'[A-Za-z_$][\w$]*',member.value,re.ASCII):
                    add(owner,cls,member.value,'method',False,len(children)-3,member,phase,
                        receiver_hint=recv.value)
        # Clojure constructor form (ClassName. args*) or (new ClassName args*).
        class_name=None;argc=None
        if h.endswith('.') and h!='.' and len(h)>1:
            class_name=_class_symbol(h[:-1],imports)
            argc=len(children)-1
        elif h=='new' and len(children)>=2 and children[1].kind=='atom':
            class_name=_class_symbol(children[1].value,imports)
            argc=len(children)-2
        if class_name:
            add(owner,class_name,'<init>','constructor',False,argc,head,phase)
        # A user macro may treat args as data, so this is only structural
        # member evidence and NEVER an executed invocation assertion.
        if head.kind != 'atom':
            walk(head,env,owner,phase,depth+1)
        for child in children[1:]:walk(child,env,owner,phase,depth+1)

    for top in forms:
        info=_definition(top)
        if not info or info[1] not in ('function','macro'):
            continue
        owner,kind,_part=info
        rest=list(top.children[2:])
        if rest and rest[0].kind=='string':rest.pop(0)
        if rest and rest[0].kind=='map':rest.pop(0)
        if rest and rest[-1].kind=='map' and rest[0].kind=='list':rest.pop()
        arities=_fn_arities(tuple(rest))
        if arities is None:
            continue
        for params,expressions in arities:
            env={}
            for param in params.children:
                if param.kind=='atom' and param.value=='&':continue
                names=binding_names(param)
                if names is None:break
                for name in names:env[name]=None
                if param.kind=='atom':env[param.value]=_tagged_binding(param,imports)
            else:
                for expression in expressions:
                    walk(expression,env,owner,
                         'macro_expansion' if kind=='macro' else 'runtime_candidate')
    return out


def analyze_jvm_interop(forms, snapshot, *, aliases, alias_guards,
                        referred, shadowed_core, macro_aliases,
                        target_jvm_major: int | None = None) -> dict:
    if not snapshot:
        return {'status':'not_requested', 'sites':[]}
    if (target_jvm_major is not None and (type(target_jvm_major) is not int or
            not 8 <= target_jvm_major <= MAX_SUPPORTED_TARGET_JAVA)):
        return {'status':'invalid_target_jvm_major', 'sites':[]}
    imports,conflicts=java_imports(forms)
    sites=member_sites(forms,imports,aliases,alias_guards,referred,
                       shadowed_core,macro_aliases)
    sites.extend(literal_typed_interop_sites(forms,imports))
    unique=[];seen=set()
    for site in sites:
        identity=(site['owner'],site['class'],site['member'],
                  site['line'],site['column'],site['member_kind'],
                  site.get('reference_kind'))
        if identity not in seen:
            seen.add(identity);unique.append(site)
    sites=unique
    classes=sorted({site['class'] for site in sites})
    if len(classes)>MAX_INTEROP_CLASSES or len(sites)>=MAX_INTEROP_SITES:
        return {'status':'bounded_evidence_unavailable',
                'reason':'interop_class_or_site_limit','sites':[],
                'java_import_conflicts':conflicts}
    verified_jars={}
    class_info={name:class_from_classpath(snapshot,name,
        target_jvm_major=target_jvm_major,verified_jars=verified_jars) for name in classes}

    def read_class(name):
        if name not in class_info:
            if len(class_info) >= MAX_HIERARCHY_CLASSES:
                return {'status': 'hierarchy_class_limit'}
            class_info[name] = class_from_classpath(snapshot,name,
                target_jvm_major=target_jvm_major,verified_jars=verified_jars)
        return class_info[name]

    def candidates_in_class(info, site):
        kind = 'fields' if site['member_kind'] == 'field' else 'methods'
        candidates = [m for m in info[kind]
                      if m['name'] == site['member'] and m['static'] == site['static']]
        if kind == 'methods':
            arity_matched = []
            for method in candidates:
                try:
                    if site.get('param_tags_unresolved_reason'):
                        continue
                    if not _param_tags_match(method,site.get('param_tag_descriptors')):
                        continue
                    if site['argc'] is None:
                        match = 'qualified_method_value_no_invocation_arity'
                    else:
                        match = _arity_compatibility(method, site['argc'])
                    if match is not None:
                        arity_matched.append({**method, 'arity_evidence': match})
                except BytecodeUnavailable:
                    continue
            candidates = arity_matched
        return candidates

    outcome=[]
    for site in sites:
        info=read_class(site['class'])
        if (site.get('qualified_static_ambiguous') and
                info.get('status')=='classfile_public_members_read'):
            has_field=any(m['name']==site['member'] and m['static']
                          for m in info['fields'])
            has_method=any(m['name']==site['member'] and m['static']
                           for m in info['methods'])
            if has_method and not has_field:
                site={**site, 'member_kind':'method'}
        base={**site,'resolution':'unresolved', 'classfile_status':info['status'],
              'call_binding_confirmed':False}
        if info['status'] == 'classfile_public_members_read' and info['public_class']:
            # Exact source class remains authoritative for its own declarations.
            # Java members can also be inherited from superclasses. Inspecting
            # only the selected class would miss genuine bytecode evidence.
            # This is still a candidate set, NOT Java overload selection.
            matches=[]
            visited=set()
            next_name=site['class']
            stop_reason='java_lang_object_root'
            depth=0
            while next_name and depth < MAX_HIERARCHY_DEPTH:
                if next_name in visited:
                    stop_reason='class_hierarchy_cycle'
                    break
                visited.add(next_name)
                parent_info=read_class(next_name)
                if parent_info['status']!='classfile_public_members_read' or not parent_info['public_class']:
                    stop_reason=('superclass_'+parent_info['status'])
                    break
                members=candidates_in_class(parent_info,site)
                if site['member_kind']=='constructor' and depth:
                    break  # JVM constructors are NEVER inherited.
                for member in members:
                    # A nearest public method or field with the same JVM
                    # descriptor shadows that exact inherited declaration.
                    # Other descriptors stay separate, unresolved candidates.
                    if any(x['descriptor']==member['descriptor'] for x in matches):
                        continue
                    matches.append({'descriptor':member['descriptor'],
                        'declaring_class':parent_info['class'],
                        'class_sha256':parent_info['sha256'],
                        'classpath_position':parent_info['classpath_position'],
                        'classpath_entry_sha256':parent_info['classpath_entry_sha256'],
                        'selected_class_resource':parent_info['selected_class_resource'],
                        'selected_multi_release_version':parent_info['selected_multi_release_version'],
                        'inherited':depth>0,
                        'varargs_declaration':member.get('varargs',False),
                        'arity_evidence':member.get('arity_evidence')})
                if site['member_kind']=='constructor':
                    stop_reason='constructor_not_inherited'
                    break
                parent=parent_info.get('superclass')
                if not parent:
                    stop_reason='java_lang_object_root'
                    break
                # `java.lang.Object` is not a license to auto-load JDK modules.
                # Missing classpath providers remain explicitly unverified.
                next_name=parent
                depth+=1
            else:
                if depth>=MAX_HIERARCHY_DEPTH:
                    stop_reason='superclass_depth_limit'
            if any(read_class(name).get('interfaces') for name in visited
                   if read_class(name).get('status')=='classfile_public_members_read'):
                # A declared superinterface member is independently verifiable
                # even when Java's maximally-specific default-method / field
                # resolution cannot be decided without a real compiler.
                # Do not conflate an interface declaration with an inherited
                # class method or claim a runtime call target.
                interface_candidates=[]
                pending=[]
                for name in sorted(visited):
                    parent_class=read_class(name)
                    if parent_class['status']=='classfile_public_members_read':
                        pending.extend((i,1) for i in parent_class['interfaces'])
                inspected=set()
                interface_note='observed_interface_declarations_inspected'
                while pending:
                    iface,level=pending.pop(0)
                    if iface in inspected:
                        continue
                    if level>MAX_INTERFACE_DEPTH:
                        interface_note='interface_depth_limit'
                        break
                    inspected.add(iface)
                    iface_info=read_class(iface)
                    if iface_info['status']!='classfile_public_members_read' or not iface_info.get('interface'):
                        interface_note='interface_'+iface_info['status']
                        break
                    for member in candidates_in_class(iface_info,site):
                        # Java interface static methods belong to the
                        # declaring interface; implementing classes do not
                        # inherit those methods (JVMS 5.4.3.3 / JLS 9.4).
                        if site['member_kind'] != 'field' and member['static']:
                            continue
                        interface_candidates.append({
                            'descriptor':member['descriptor'],
                            'declaring_class':iface_info['class'],
                            'class_sha256':iface_info['sha256'],
                            'classpath_position':iface_info['classpath_position'],
                            'classpath_entry_sha256':iface_info['classpath_entry_sha256'],
                            'selected_class_resource':iface_info['selected_class_resource'],
                            'selected_multi_release_version':iface_info['selected_multi_release_version'],
                            'inherited':True,
                            'origin_kind':'superinterface_declaration_unselected',
                            'varargs_declaration':member.get('varargs',False),
                            'arity_evidence':member.get('arity_evidence')})
                    pending.extend((i,level+1) for i in iface_info['interfaces'])
            else:
                interface_candidates=[]
                interface_note='no_declared_interfaces_in_inspected_classes'
            inherited=any(m['inherited'] for m in matches)
            kind='fields' if site['member_kind']=='field' else 'methods'
            declared=any(not m['inherited'] for m in matches)
            if matches:
                if site.get('param_tag_descriptors') is not None:
                    resolved='classfile_param_tag_signature_candidate_not_compiler_selection'
                elif site.get('reference_kind')=='qualified_member_value' and kind=='methods':
                    resolved='classfile_qualified_method_value_candidate_not_compiler_selection'
                elif (kind=='methods' and all(m.get('arity_evidence') ==
                                            'varargs_expansion_candidate' for m in matches)):
                    resolved=('classfile_public_inherited_varargs_arity_candidate'
                              if inherited and not declared else
                              'classfile_public_varargs_arity_candidate')
                elif inherited and not declared:
                    resolved=('classfile_public_inherited_method_name_and_arity_exists'
                              if kind=='methods' else 'classfile_public_inherited_field_exists')
                elif site['member_kind']=='constructor':
                    resolved='classfile_public_constructor_arity_exists'
                elif kind=='methods':
                    resolved='classfile_public_method_name_and_arity_exists'
                else:
                    resolved=('classfile_public_static_field_exists' if site['static']
                              else 'classfile_public_instance_field_exists')
            else:
                resolved=('param_tags_unresolved_no_signature_claim' if site.get('param_tags_unresolved_reason')
                          else 'member_not_declared_in_selected_classfile')
            base.update(classpath_position=info['classpath_position'],
                        classpath_entry_basename=info['classpath_entry_basename'],
                        classpath_entry_sha256=info['classpath_entry_sha256'],
                        class_sha256=info['sha256'],
                        class_major_version=info['major_version'],
                        selected_class_resource=info['selected_class_resource'],
                        selected_multi_release_version=info['selected_multi_release_version'],
                        target_jvm_major_user_asserted=target_jvm_major,
                        multi_release_jar=info['multi_release_jar'],
                        candidate_descriptors=sorted({m['descriptor'] for m in matches})[:24],
                        candidate_declarations=sorted(matches,key=lambda m:(m['inherited'],m['declaring_class'],m['descriptor']))[:24],
                        superinterface_declarations=sorted(interface_candidates, key=lambda m:(m['declaring_class'],m['descriptor']))[:24],
                        superinterface_candidate_descriptors=sorted({m['descriptor'] for m in interface_candidates})[:24],
                        superclass_chain_checked=sorted(visited),
                        superclass_lookup_status=stop_reason,
                        interface_lookup_status=interface_note,
                        resolution=(('classfile_public_superinterface_member_declared_unresolved_selection'
                                     if interface_candidates else resolved)
                                    if not matches else resolved),
                        single_declaration_in_this_class=sum(not x['inherited'] for x in matches)==1,
                        overload_identity_proven=False,
                        call_binding_confirmed=False)
        outcome.append(base)
    return {'status':'checked_explicit_classpath_bytecode_not_compiler_calls',
            'site_count':len(outcome), 'sites':outcome,
            'target_jvm_major_user_asserted': target_jvm_major,
            'java_import_conflicts':conflicts,
            'limits':{'max_classes':MAX_INTEROP_CLASSES,
                      'max_sites':MAX_INTEROP_SITES,
                      'max_superclass_depth':MAX_HIERARCHY_DEPTH,
                      'max_hierarchy_classes':MAX_HIERARCHY_CLASSES,
                      'max_interface_depth':MAX_INTERFACE_DEPTH},
            'limitations':['superinterface_declarations_not_method_selection','argument_types_not_inferred',
                           'compiler_overload_binding_not_executed',
                           'varargs_expansion_is_candidate_not_compiler_selection',
                           'qualified_method_values_not_compiler_selected',
                           'param_tags_match_only_explicit_bytecode_descriptors',
                           'macros_not_expanded','JDK_modules_not_automatically_loaded',
                           'target_jvm_version_user_asserted_not_compiler_verified',
                           'classpath_files_are_not_an_atomic_filesystem_snapshot']}
