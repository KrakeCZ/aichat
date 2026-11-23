"""Source-only JVM require prefix-list normalization, never library loading."""
from dataclasses import replace


def simple_symbol(form):
    """Conservative unqualified reader symbol slot; literals are not symbols."""
    if form.kind!='atom':
        return False
    name=form.value
    return bool(name and not name.startswith((':','\\','#')) and '/' not in name
                and not name.endswith(':') and '::' not in name
                and name not in {'nil','true','false','&'} and not name[0].isdigit()
                and not (name[0] in '+-.' and len(name)>1 and name[1].isdigit()))


def expand_prefix(spec, validate):
    """Return all flat specs or None. One invalid child rejects the whole group.

    Official require prefix lists are sequential. Nested lib names may not
    contain periods; nested prefix lists are not flattened recursively.
    The replacement namespace keeps the child token's source coordinates.
    """
    if spec.kind not in ('list','vector') or len(spec.children)<2:
        return None
    prefix=spec.children[0]
    if not simple_symbol(prefix):
        return None
    expanded=[]
    for child in spec.children[1:]:
        if child.kind=='atom':
            target=child
        elif child.kind=='vector' and child.children:
            target=child.children[0]
        else:
            return None
        if not simple_symbol(target) or '.' in target.value:
            return None
        name=replace(target,value=prefix.value+'.'+target.value)
        normalized=(name if child.kind=='atom' else
                    replace(child,children=(name,*child.children[1:])))
        if validate(normalized,'clj'):
            return None
        expanded.append(normalized)
    return expanded


def expanded_require_forms(forms, platform, validate):
    """Normalize only literal :require in JVM ns forms; preserve other forms."""
    if platform!='clj':
        return forms
    output=[]
    for form in forms:
        if form.kind!='list' or not form.children or form.children[0].value!='ns':
            output.append(form);continue
        clauses=[]
        for clause in form.children[2:]:
            if (clause.kind!='list' or not clause.children or
                    clause.children[0].value!=':require'):
                clauses.append(clause);continue
            specs=[]
            for spec in clause.children[1:]:
                # core/libspec?: symbols and vectors with absent/keyword
                # second element are flat specs, all other sequences prefixes.
                prefix=(spec.kind=='list' or (spec.kind=='vector' and
                    len(spec.children)>1 and not spec.children[1].value.startswith(':')))
                expanded=expand_prefix(spec,validate) if prefix else None
                specs.extend(expanded if expanded is not None else (spec,))
            clauses.append(replace(clause,children=(clause.children[0],*specs)))
        output.append(replace(form,children=(*form.children[:2],*clauses)))
    return output


def invalid_prefix_aliases(forms, platform, validate):
    """Unsupported prefix groups still block syntactically conflicting aliases."""
    if platform!='clj':return set()
    aliases=set()
    def collect(form):
        if form.kind=='vector':
            for i in range(1,len(form.children)-1,2):
                if form.children[i].value in {':as',':as-alias'} and simple_symbol(form.children[i+1]):
                    aliases.add(form.children[i+1].value)
        for child in form.children:collect(child)
    for form in forms:
        if form.kind!='list' or not form.children or form.children[0].value!='ns':continue
        for clause in form.children[2:]:
            if clause.kind!='list' or not clause.children or clause.children[0].value!=':require':continue
            for spec in clause.children[1:]:
                prefix=(spec.kind=='list' or (spec.kind=='vector' and len(spec.children)>1
                       and not spec.children[1].value.startswith(':')))
                if prefix and expand_prefix(spec,validate) is None:collect(spec)
    return aliases
