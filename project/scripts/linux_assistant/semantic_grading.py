"""Check explicit output format only; content/steps require documented semantic review."""
import json
import re
import shlex

def _json_type_matches(value,spec):
    """Check field types without imposing the reference answer's wording."""
    if isinstance(spec,str):spec={'type':spec}
    kind=spec['type']
    types={'string':str,'boolean':bool,'integer':int,'object':dict,'array':list,'null':type(None)}
    if kind=='number':return type(value) in (int,float)
    if kind not in types:raise ValueError('Unsupported JSON type contract: '+kind)
    if type(value) is not types[kind]:return False
    if kind=='array' and 'items' in spec:
        return all(_json_type_matches(item,spec['items']) for item in value)
    return True


def format_check(answer,spec):
    kind=spec.get('kind','free')
    if not answer.strip():return False,'empty answer'
    if kind=='free':return True,None
    if kind=='exact':
        value=answer.strip() if spec.get('allow_outer_whitespace',False) else answer
        return (value==spec['expected'],None if value==spec['expected'] else 'exact output differs')
    if kind=='json':
        try:value=json.loads(answer)
        except (ValueError,TypeError):return False,'not a standalone JSON document'
        expected=spec['expected']
        ok=type(value) is dict and value.keys()==expected.keys() and all(type(value[k]) is type(v) and value[k]==v for k,v in expected.items())
        return ok,None if ok else 'JSON fields, values or types differ'
    if kind=='json_schema':
        try:value=json.loads(answer)
        except (ValueError,TypeError):return False,'not a standalone JSON document'
        fields=spec['fields']
        ok=(type(value) is dict and set(fields)<=set(value) and
            (spec.get('additional_fields',False) or value.keys()==fields.keys()) and
            all(_json_type_matches(value[key],expected) for key,expected in fields.items()))
        return ok,None if ok else 'JSON field names or types differ; content is separately reviewed'
    if kind=='command_only':
        lines=answer.strip().splitlines()
        ok=len(lines)==1 and not any(s in answer for s in ('```','~~~~'))
        if ok:
            try:words=shlex.split(lines[0],comments=False,posix=True)
            except ValueError:words=[]
            # Keep the first word recognizable as a shell command token. This
            # catches explanations such as “请运行 ls -la。” without pretending
            # to be a complete shell parser or syntax validator.
            ok=bool(words) and re.fullmatch(r'[A-Za-z0-9_./~+-]+',words[0]) is not None
        return ok,None if ok else 'expected one command line without wrapper'
    if kind=='lines':
        ok=len(answer.strip().splitlines())==spec['count'] and not any(s in answer for s in ('```','~~~~'))
        return ok,None if ok else 'line count or wrapper differs'
    if kind=='items':
        labels=spec['labels'];positions=[]
        for label in labels:
            matches=list(re.finditer(re.escape(label)+r'\s*[：:]',answer))
            if len(matches)!=1:return False,'expected each requested item label once'
            positions.append(matches[0].start())
        ok=positions==sorted(positions)
        return ok,None if ok else 'requested item order differs'
    if kind=='numbered_items':
        numbers=re.findall(r'(?:^|\s)([1-9])(?:[.、．)]|）)\s*',answer)
        ok=numbers==[str(i) for i in range(1,spec['count']+1)]
        return ok,None if ok else 'requested numbered items differ'
    if kind=='sentence':
        ok=len([p for p in re.split('[。！？!?]',answer.strip()) if p.strip()])<=1 and len(answer.strip().splitlines())==1
        return ok,None if ok else 'expected one sentence'
    raise ValueError('Unknown format kind: '+kind)

def validate_model_review(row,decision):
    """No single accepted flag may replace the three independent review axes."""
    grading=row['grading'];axes=decision.get('axes',{})
    completeness=axes.get('completeness',axes.get('steps_complete'))
    if ('completeness' in axes and 'steps_complete' in axes and
            axes['completeness']!=axes['steps_complete']):
        raise ValueError('Completeness aliases conflict: '+row['id'])
    if (type(axes.get('content_correct')) is not bool or type(completeness) is not bool or
            type(axes.get('format_compliant')) is not bool):
        raise ValueError('New data requires explicit content/steps/format decisions: '+row['id'])
    actual_format,_=format_check(row['answer'],grading['format'])
    # A machine failure is binding. Passing a partial format check (e.g. one
    # line) cannot force a reviewer to accept prose as a command-only answer.
    if not actual_format and axes['format_compliant']:raise ValueError('Review conflicts with explicit format: '+row['id'])
    required_snippets=grading.get('content',{}).get('required_snippets',[])
    missing=[snippet for snippet in required_snippets if snippet not in row['answer']]
    if missing and axes['content_correct']:
        raise ValueError('Review marks source-required text correct although it is missing: '+row['id'])
    expected=axes['content_correct'] and completeness and axes['format_compliant']
    if type(decision.get('accepted')) is not bool or decision['accepted']!=expected:
        raise ValueError('Accepted flag conflicts with review axes')
    return {**axes,'completeness':completeness}
