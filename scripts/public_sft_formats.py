"""Formatting-only adapters for pinned public dialogue archives."""

def strip_moss_tag(text,role,end):
    if not isinstance(text,str):raise ValueError('invalid tag content')
    text=text.strip();start='<|'+role+'|>:'
    if not text.startswith(start) or not text.endswith(end):raise ValueError('unexpected MOSS framing')
    return text[len(start):-len(end)].strip()

def moss_dialogue(row):
    # No tool outputs, thought traces, role substitutions, or generated answers.
    try:
        ms=[]
        for n in range(1,row['num_turns']+1):
            turn=row['chat']['turn_'+str(n)]
            if any(strip_moss_tag(turn[k],role,end)!='None' for k,role,end in [('Commands','Commands','<eoc>'),('Tool Responses','Results','<eor>'),('Inner Thoughts','Inner Thoughts','<eot>')]):return {}
            ms.extend([dict(role='user',content=strip_moss_tag(turn['Human'],'Human','<eoh>')),dict(role='assistant',content=strip_moss_tag(turn['MOSS'],'MOSS','<eom>'))])
        # The published MOSS-specific meta_instruction is replaced by the same
        # LubanLM system template used for the other public instruction sources.
        return dict(conversations=ms)
    except (KeyError,TypeError,ValueError):return {}
