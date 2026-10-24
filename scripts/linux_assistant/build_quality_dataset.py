"""Rebuild the single active dataset from independent cases and explicit semantic criteria.

All earlier inputs/acceptance files are archived byte-for-byte before replacement.
No model is trained or evaluated by this entry point.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from scripts.common import WORKSPACE,sha256,write_json
from scripts.linux_assistant.common import jsonl,write_jsonl,fingerprint
from scripts.linux_assistant.quality_cases import PAIRS,EVAL_CASES,SOURCES
from scripts.linux_assistant.semantic_grading import format_check

DATA=WORKSPACE/'datasets/linux-assistant-unified'
REPORT=WORKSPACE/'reports/linux-assistant-data-quality'
LABELS={'environment':'环境检查','linux':'Linux操作','deployment':'部署流程','troubleshooting':'故障排查','interaction':'交互规范','general':'通用能力'}
SYSTEM_PROMPT='你是中文助手，擅长Linux与开发板部署。依据题目和已知信息回答，普通常识、算术和推理问题按题意直接作答。操作建议说明必要步骤和判断依据；只有缺少影响结论的关键条件时才提出具体追问。区分事实与推测，不编造设备状态、命令输出、版本或SDK接口。遵守用户明确指定的格式与最新更正。你只提供建议，不执行命令。'
DROP_META={'interaction-unknownboard-00','interaction-unseenoutput-00','interaction-uncertain-00',
    'interaction-preservecontext-00','interaction-steps-00','interaction-secret-00','interaction-noexecute-00'}

def snapshot_inputs():
    control=DATA/'curation/build-inputs.json'
    if control.exists():
        value=json.loads(control.read_text('utf8'));root=WORKSPACE/value['snapshot']
        for name,digest in value['files'].items():
            if sha256(root/name)!=digest:raise ValueError('Archived original changed: '+name)
        return root,value
    digest=sha256(DATA/'manifest.json');root=WORKSPACE/'provenance/dataset-snapshots'/('before-quality-'+digest[:16])
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    # Never move/delete prior sources; preserve their exact bytes for old experiments.
    for p in DATA.rglob('*'):
        if p.is_file():
            dest=root/p.relative_to(DATA);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    files={str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in root.rglob('*') if p.is_file()}
    value={'snapshot':str(root.relative_to(WORKSPACE)).replace('\\','/'),'files':files,
        'previous_manifest_sha256':digest,'policy':'Preserve original acceptance bytes; active development and scoring are rebuilt before any new model evaluation.'}
    write_json(control,value)
    return root,value

def question(row):return next(m['content'] for m in reversed(row['messages']) if m['role']=='user')

def inferred_format(q,answer):
    if '只输出' in q and '原文' in q:return {'kind':'exact','expected':answer}
    if '只输出JSON' in q or ('只输出' in q and 'JSON' in q) or '只输出同样两字段JSON' in q:
        return {'kind':'json','expected':json.loads(answer)}
    if '只输出' in q and ('结果' in q or '数值' in q or '数' in q and '命令' not in q):
        if '\n' not in answer and len(answer)<35:return {'kind':'exact','expected':answer}
    if '两行' in q and any(s in q for s in ('只输出','只给','不要说明','不要解释')):
        return {'kind':'lines','count':2}
    if ('只输出' in q or '只给' in q or '不加解释' in q) and '命令' in q:
        return {'kind':'command_only'}
    if any(s in q for s in ('只用一句话','用一句话说明','用一句话解释')):return {'kind':'sentence'}
    if '三项格式' in q or '按' in q and '三项' in q:return {'kind':'items','labels':['检查','判断','下一步']}
    if '三个编号检查项' in q:return {'kind':'numbered_items','count':3}
    return {'kind':'free'}

def atomic_facts(answer):
    parts=[p.strip() for p in re.split('[。；]',answer) if p.strip()]
    # Only the direct answer is compulsory; additional cautions are not hidden tests.
    return parts[:2] if parts and len(parts[0])<8 else parts[:1]

def row(category,group,case_id,context,q,answer,required,sources,split='train',multi=False,fmt=None,origin='authored_synthetic_fixture'):
    content=(context+'\n'+q).strip() if not multi else q
    messages=([{'role':'user','content':context},{'role':'assistant','content':'已记录这些已知条件；下面根据它们继续。'}] if multi else [])
    messages.extend([{'role':'user','content':content},{'role':'assistant','content':answer}])
    spec=fmt or inferred_format(q,answer)
    if spec['kind']=='exact':spec={**spec,'allow_outer_whitespace':'原文' not in q}
    # Sources for elementary questions are their self-contained givens, not an
    # unverifiable claim of external authority.
    sources=['problem://given-facts' if s=='authored elementary task' else s for s in sources]
    if group in ('environment/nproc-scope','environment/nproc'):
        sources=list(dict.fromkeys([SOURCES['nproc'][1],*sources]))
    item={'id':f'{split}-{category}-{case_id}','group':group,'case_id':case_id,'kind':'general' if category=='general' else 'task',
        'category':category,'category_label':LABELS[category],'messages':messages,'reference':answer,'rubric':'',
        'facts':required,'sources':sources,'split':split,'scenario_origin':origin,
        'grading':{'policy':'semantic_content_steps_explicit_format','content':{'required_facts':required,
            'optional_facts':[],'forbidden_claims':['无证据声称已执行或访问设备','未提供的设备状态、版本或SDK接口']},
            'steps':{'required':[],'applicable':False},'format':spec,
            'alternatives_policy':'Accept any factually valid method fulfilling the stated task and constraints; do not require reference wording or optional cautions.',
            'scoring':'Review content_correct, steps_complete and format_compliant separately; necessary facts must be entailed by the question or indispensable to the task.'}}
    ok,reason=format_check(answer,spec)
    if not ok:raise ValueError('Reference format mismatch: '+item['id']+' '+str(reason))
    if group=='interaction/test-format-steps':
        item['grading']['steps']={'applicable':True,'required':['先提出只读路径及错误检查','据检查对象解释可能原因','索取启动命令、路径和原始错误用于下一步判断']}
    if group in ('troubleshooting/partial','deployment/checksum'):
        item['grading']['steps']={'applicable':True,'required':['在尝试加载之前核对完整文件SHA256清单','摘要不符时不继续把该制品作为完整模型部署']}
    return item

def convert_seed(original,split):
    r=copy.deepcopy(original);msgs=r['messages'][:-1]
    if len(msgs)>2 and msgs[0]['content']=='我在学习Linux与开发板部署，请用中文帮助我核对问题。':msgs=msgs[2:]
    q=msgs[-1]['content']
    for prefix in ('请给出准确而简洁的解释：','我在整理部署笔记，请回答：','面向Linux初学者回答：','请说明操作依据：',
        '请帮我确认：','我需要核对这一步：','请指出关键检查项：','请按实际可确认的信息回答：','请帮我复核下面的问题：'):
        if q.startswith(prefix):q=q[len(prefix):]
    msgs[-1]['content']=q
    # Semantic duplicates missed by ID/group-only checks in earlier data.
    replacements={
      'troubleshooting/dev-shell-status':('troubleshooting/rename-exdev',
        '程序调用rename把文件从一个文件系统移到另一个，返回EXDEV。应先核对什么，改文件权限能消除这个边界吗？',
        '先核对源和目标所属文件系统。rename不能跨该文件系统边界；改权限不会消除EXDEV。需要跨盘迁移时应另行复制并校验，再按明确授权处理源文件，不把它当作同一次原子重命名。',SOURCES['rename'][1]),
      'environment/test-process-status':('environment/zombie-process-state',
        '允许读取的/proc/5678/status中State为Z (zombie)。这表示进程还在执行模型计算吗？',
        '不表示。Z是僵尸状态，进程已经终止但父进程尚未回收其退出信息；应核对父进程和回收情况，不能把它记成正在计算。',SOURCES['proc-status'][1]),
      'linux/diskinode':('linux/quoted-hash-literal',
        '在Bash执行printf "%s\\n" "model#1"。引号内的#会开始注释吗？只输出这一行结果。',
        'model#1',SOURCES['bash'][1]),
      'interaction/given-scope-only':('interaction/literal-special-characters',
        '只输出下面这段原文，不加引号、代码块或转义字符：a[b]\\c',
        'a[b]\\c','problem://given-facts'),
    }
    if r['group'] in replacements:
        r['group'],q,r['reference'],source=replacements[r['group']]
        r['sources']=[source];r['rubric']='';msgs=[{'role':'user','content':q}]
    if r['group']=='environment/effective-uid':
        q='怎样查看当前调用进程的有效用户ID？';msgs[-1]['content']=q
        r['reference']='运行id -u查看有效UID；它不代表其他服务进程的身份，也不足以单独判断文件访问权限。'
    if r['group']=='environment/swap-areas':
        q='如何列出系统当前已启用的交换区？';msgs[-1]['content']=q
    if r['group']=='linux/copy':
        r['reference']='可用cp -a 源目录 目标目录；先确认目标是否存在及期望层级。元信息保留还受执行权限和文件系统支持限制。'
    if r['group']=='deployment/libs':
        msgs[-1]['content']='项目记录Toolkit和Runtime均为1.3.1。除此以外，板端还应核对什么兼容信息？'
        q=msgs[-1]['content'];r['reference']='还需记录NPU驱动和实际加载库，按固定SDK说明核对兼容关系，并实际加载验证；两个组件版本相同不等于所有组合都兼容。'
    if r['group']=='deployment/bf16w4':
        msgs[-1]['content']='本题已知PC训练使用BF16 LoRA，板端导出W4A16。报告应把两个阶段精度混写成4位训练吗？'
        q=msgs[-1]['content'];r['reference']='不应。训练精度记录BF16 LoRA，部署量化记录W4A16，两阶段分开。'
    if r['id']=='troubleshooting-decode-error-preserve-evidence-02':
        msgs[-1]['content']='日志预期编码为UTF-8且文件已写完，但某序列无效，能仅凭报错断言设备内存损坏吗？';q=msgs[-1]['content']
    if r['group']=='interaction/test-format-steps':
        q='只允许只读检查。程序使用绝对配置路径但打不开文件，尚未给报错。请按“检查、判断、下一步”恰好三项回答。'
        msgs[-1]['content']=q
        r['reference']='检查：核对配置路径对应对象和实际错误输出。\n判断：绝对路径仍可能不存在、类型不符或受访问限制。\n下一步：请提供启动命令、配置路径和完整报错。'
    if r['group']=='interaction/clarifyversion':
        q='在RKLLM部署排查中，用户只说“SDK最新版”，如何询问实际兼容组合？';msgs[-1]['content']=q
    if r['group']=='deployment/vision':
        q='本题已知板端只部署语言分支，没有部署视觉编码器。文字问答测试能证明看图可用吗？';msgs[-1]['content']=q
    if r['group']=='deployment/contextmodel':
        q='本题板端导出上下文为2048 Token，原模型上游支持更长上下文。板端会自动支持相同长度吗？';msgs[-1]['content']=q
    answer=r['reference'];facts=atomic_facts(answer)
    fmt=inferred_format(q,answer)
    if r['rubric'].startswith('JSON:'):fmt={'kind':'json','expected':json.loads(answer)}
    if r['rubric'].startswith('EXACT:'):fmt={'kind':'exact','expected':answer}
    if 'MAXLINES:1' in r['rubric']:fmt={'kind':'sentence'}
    # Well-known evaluator defects: judge requested behavior, not arbitrary extra tokens.
    overrides={
      'linux/whichpython':['使用所需Python解释器的-m pip调用对应pip'],
      'linux/exitcode':['紧接上一命令读取$?，如echo $?或printf'],
      'deployment/dev-bounded-smoke':['给出有效的运行超时限制方法；题目未指定秒数，不强制30s'],
      'linux/dev-resolve-symlink':['解析到最终规范路径，接受readlink -f或realpath等有效方法'],
      'environment/cpuinfo':['给出读取内核CPU信息的有效方法；不把架构免责声明当必答关键词'],
      'environment/uptime':['有效查询系统自启动时长和负载，不混淆进程时间或CPU百分比'],
      'environment/dev-memavailable':['优先MemAvailable这一可用估计，不把MemFree当等同指标'],
    }
    facts=overrides.get(r['group'],facts)
    if split=='train' and r['group'] in ('interaction/preservecontext','interaction/uncertain') and '应' in answer[:4]:
        raise ValueError('Reviewer-style meta target must be retired')
    clean=row(r['category'],r['group'],fingerprint({'group':r['group'],'messages':msgs})[:14],
        '',q,answer,facts,r['sources'],split=split,fmt=fmt,origin='reviewed_legacy_seed')
    clean['messages']=[*msgs,{'role':'assistant','content':answer}]
    clean['legacy_id']=r['id']
    return clean

def general_training():
    records=[];rng=random.Random(20261020)
    # These inputs are new, distinct from the previous development/acceptance arithmetic.
    for op in ('add','sub','mul','div'):
        for i in range(20):
            x=31+i*3 if op!='mul' else 12+i
            y=9+i%7 if op!='div' else 3+i%5
            if op=='mul':y=2+i%6
            if op=='add':answer=str(x+y);expr=f'{x}+{y}';symbol='+'
            elif op=='sub':answer=str(x-y);expr=f'{x}-{y}';symbol='-'
            elif op=='mul':answer=str(x*y);expr=f'{x}×{y}';symbol='×'
            else:x=(14+i)*y;answer=str(x//y);expr=f'{x}÷{y}';symbol='÷'
            multi=i%2==1
            ctx=f'本题给定A={x}、B={y}，运算为{symbol}。' if multi else ''
            q=f'只输出A{symbol}B的结果。' if multi else f'只输出{expr}的结果。'
            r=row('general',f'general/{op}/{x}-{y}',f'{op}-{x}-{y}',ctx,q,answer,[f'{expr}={answer}'],
                ['problem://given-facts'],multi=multi,fmt={'kind':'exact','expected':answer})
            r['oracle']={'kind':'arithmetic','operation':op,'a':x,'b':y,'result':int(answer)};records.append(r)
    for i in range(20):
        count=17+i;values=dict(zip(('A','B','C'),rng.sample([count,count+3,count-2],3)))
        ctx='三件物品的长度分别为：'+ '，'.join(f'{k}={v}厘米' for k,v in values.items())+'。'
        answer=(max if i%2==0 else min)(values,key=values.get);q='只输出最长物品的代号。' if i%2==0 else '只输出最短物品的代号。'
        r=row('general',f'general/order/{i}',f'order-{i}',ctx,q,answer,[f'正确比较三个给定长度，答案{answer}'],
            ['problem://given-facts'],multi=i%2==1,fmt={'kind':'exact','expected':answer})
        r['oracle']={'kind':'order','values':values,'select':'max' if i%2==0 else 'min'};records.append(r)
    for i in range(20):
        value={'count':37+i,'ready':i%2==0,'runtime':None if i%3==0 else f'example-{i}'}
        q='仅输出JSON对象，count为整数'+str(value['count'])+'，ready为布尔'+str(value['ready']).lower()+'，runtime为'+('null' if value['runtime'] is None else '字符串'+value['runtime'])+'；不要额外字段或代码块。'
        ans=json.dumps(value,ensure_ascii=False,separators=(',',':'))
        r=row('general',f'general/typed-json/{i}',f'typed-json-{i}','',q,ans,['精确字段、值与JSON类型'],
            ['problem://given-facts'],fmt={'kind':'json','expected':value});r['oracle']={'kind':'json','expected':value};records.append(r)
    for i in range(20):
        ctx=f'清单有{73+i}个盒子，每盒{11+i%6}个零件，全部完整。';answer=str((73+i)*(11+i%6))
        r=row('general',f'general/word-count/{i}',f'word-count-{i}',ctx,'只输出零件总数。',answer,['按给定盒数乘每盒数量'],
            ['problem://given-facts'],multi=True,fmt={'kind':'exact','expected':answer})
        r['oracle']={'kind':'arithmetic','operation':'mul','a':73+i,'b':11+i%6,'result':int(answer)};records.append(r)
    for i in range(20):
        values=[3+i,8+i,13+i,21+i];ans=str(sum(values))
        ctx='本题列表为'+json.dumps(values)+'。'
        r=row('general',f'general/list-total/{i}',f'list-total-{i}',ctx,'只输出列表元素之和。',ans,['正确求给定列表之和'],
            ['problem://given-facts'],multi=i%2==0,fmt={'kind':'exact','expected':ans})
        r['oracle']={'kind':'sum','values':values};records.append(r)
    if len(records)!=160:raise ValueError('Expected 160 deterministic general training cases')
    for r in records:
        kind=r['group'].split('/')[1]
        r['general_type']='numeric-order' if kind=='order' else kind
        r['group']='general/train/'+r['general_type']
    return records

def build():
    raise RuntimeError('This authored-case builder is retired. It must not recreate the removed dataset or its self-review artifacts.')
    archived,provenance=snapshot_inputs();REPORT.mkdir(parents=True,exist_ok=True)
    original=jsonl(archived/'train.jsonl');seen=set();retired=[];train=[]
    for old in original:
        key=(old['group'],old['reference'])
        if old['group']=='troubleshooting/service-state' or old['id'] in DROP_META or key in seen:
            reason=('service-status family excluded conservatively while holding out service diagnostics' if old['group']=='troubleshooting/service-state'
                else 'reviewer-style meta target replaced by an actual assistant response' if old['id'] in DROP_META
                else 'same-family/same-reference prompt-prefix variant collapsed')
            retired.append({'id':old['id'],'reason':reason,'original_preserved':True});seen.add(key);continue
        seen.add(key);train.append(convert_seed(old,'train'))
    for cat,family,keys,a,b in PAIRS:
        for j,case in enumerate((a,b),1):
            ctx,q,ans,required=case
            train.append(row(cat,cat+'/'+family,f'contrast-{family}-{j}',ctx,q,ans,required,[SOURCES[k][1] for k in keys],multi=j==2))
    train+=general_training()
    splits={'train':train}
    for split in ('dev','test'):
        seen=set();selected=[]
        for old in jsonl(archived/f'{split}.jsonl'):
            if old['group'] in seen:continue
            seen.add(old['group']);selected.append(convert_seed(old,split))
        for s,cat,slug,ctx,q,ans,required,keys in EVAL_CASES:
            if s==split:selected.append(row(cat,cat+'/'+slug,'heldout-'+slug,ctx,q,ans,required,
                [SOURCES[k][1] for k in keys.split(',')],split=split,multi=bool(ctx) and cat=='interaction'))
        splits[split]=selected
        general=[]
        for i,old in enumerate(jsonl(archived/f'general-{split}.jsonl')):
            r=convert_seed(old,split)
            # Four duplicate relation paraphrases become genuinely different constraints.
            if 21<=i<=24:
                names=(['晓晨','晓雨','晓星'] if split=='dev' else ['晨舟','雨舟','星舟'])
                order=[names[(i+shift)%3] for shift in range(3)]
                smallest=i%2==0;answer=order[-1] if smallest else order[0]
                q=f'{order[0]}比{order[1]}快，{order[1]}比{order[2]}快。只输出'+('最慢' if smallest else '最快')+'者的名字。'
                # A different fourth contender changes the actual maximum, not just names.
                if i>=23:q=f'另有{names[0]}的哥哥比这三人都快。'+q;answer=names[0]+'的哥哥' if not smallest else order[-1]
                r=row('general',f'general/{split}/relation-{i}',f'relation-{i}','',q,answer,['依据传递关系及题内新增条件选择目标'],
                    ['problem://given-facts'],split=split,fmt={'kind':'exact','expected':answer})
            r['general_type']=('add','mul','literal-copy','typed-json','relation-order','commonsense','word-count','sub')[i//5]
            r['group']='general/'+split+'/'+r['general_type']
            general.append(r)
        splits['general-'+split]=general
    # Programmatic examples refer only to given observations. Do not claim they ran.
    allrows=[r for records in splits.values() for r in records]
    for r in allrows:
        r['source_review']='agent-source-and-criteria-review'
        r['review_status']='source_reviewed_release_requires_hash_bound_quality_approval'
    groups={s:{r['group'] for r in splits[s]} for s in ('train','dev','test')}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        overlap=groups[a]&groups[b]
        if overlap:raise ValueError('Split family overlap: '+str(sorted(overlap)))
    ids=[r['id'] for r in allrows]
    # Full context is the actual example identity, not just the final follow-up sentence.
    inputs=[fingerprint(r['messages'][:-1]) for r in allrows]
    if len(ids)!=len(set(ids)) or len(inputs)!=len(set(inputs)):
        duplicates={}
        for r,k in zip(allrows,inputs):duplicates.setdefault(k,[]).append(r['id'])
        raise ValueError('Duplicate IDs or full input contexts: '+str([v for v in duplicates.values() if len(v)>1]))
    for r in allrows:
        if not r['facts'] or r['messages'][-1]['content']!=r['reference']:raise ValueError('Missing target/criteria')
        if any('请给出准确而简洁的解释：' in m['content'] for m in r['messages']):raise ValueError('Prefix padding retained')
        if not format_check(r['reference'],r['grading']['format'])[0]:raise ValueError('Reference format invalid: '+r['id'])
    stage=WORKSPACE/'.cache/linux-assistant-data-build';stage.mkdir(parents=True,exist_ok=True)
    for split,records in splits.items():write_jsonl(stage/(split+'.jsonl'),records)
    reviews=[]
    for split in splits:
        for group in sorted({r['group'] for r in splits[split]}):
            batch=[r for r in splits[split] if r['group']==group];refs=list(dict.fromkeys(r['reference'] for r in batch))
            reviews.append({'group':group,'split':split,'review_status':'agent_content_review','reviewer':'Codex source/criteria review; not independent human expert',
                'reference':refs[0],'source_urls':sorted({x for r in batch for x in r['sources']}),
                'evidence':{'reference_sha256':hashlib.sha256(refs[0].encode()).hexdigest(),
                    'reference_hashes':[hashlib.sha256(x.encode()).hexdigest() for x in refs],
                    'independent_review_completed':False,'runtime_verification':'tracked separately, not assumed'}})
    write_jsonl(stage/'source-review.jsonl',reviews)
    # All content checks above finish before replacing active data. Publication is
    # explicit and invalidates every old hash-bound training approval/baseline.
    for name in [*(s+'.jsonl' for s in splits),'source-review.jsonl']:
        shutil.copy2(stage/name,DATA/(name+'.building'));(DATA/(name+'.building')).replace(DATA/name)
    write_json(DATA/'curation/retired-records.json',retired)
    (DATA/'curation/system-prompt.txt').write_text(SYSTEM_PROMPT,encoding='utf8')
    write_json(DATA/'curation/source-registry.json',{k:{'title':v[0],'source':v[1],'scope':v[2]} for k,v in SOURCES.items()})
    write_json(DATA/'curation/quality-contract.json',{'construction':'structured_semantic_cases','previous_inputs':provenance,
        'no_prefix_augmentation':True,'new_authored_task_cases':150,'deterministic_general_training':160,
        'development_changed':True,'requires_new_baseline':True,'previous_results_not_comparable':True,
        'acceptance':'Active acceptance rebuilt and sealed before any new model run; previous original acceptance bytes remain archived.',
        'training':'No training launched by this dataset work. Release requires native audit, runtime fixtures, three-axis grading and record review.'})
    files={str(p.relative_to(DATA)).replace('\\','/'):sha256(p) for p in DATA.glob('*.jsonl')}
    for name in ('build-inputs.json','source-registry.json','quality-contract.json','retired-records.json','system-prompt.txt','assistant-policy.md'):
        files['curation/'+name]=sha256(DATA/'curation'/name)
    counts={s:len(r) for s,r in splits.items()};scenes={s:len({r['group'] for r in records}) for s,records in splits.items()}
    manifest={'dataset':'linux-assistant-unified','construction':'structured_semantic_cases','status':'frozen_release_requires_hash_bound_quality_approval',
        'files':files,'record_counts':counts,'scenario_counts':scenes,'total_records':sum(counts.values()),
        'category_labels':LABELS,'category_summary':{s:{LABELS[c]:{'records':sum(r['category']==c for r in records),
            'scenarios':len({r['group'] for r in records if r['category']==c})} for c in sorted({r['category'] for r in records})} for s,records in splits.items()},
        'multi_turn_training_records':sum(len(r['messages'])>2 for r in train),'unique_training_references':len({r['reference'] for r in train}),
        'task_split_group_overlap':{a+'_'+b:len(groups[a]&groups[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]},
        'source_review_records':len(reviews),'new_baseline_required':True,'final_test_policy':'sealed; no model generations until a new candidate and scoring contract are locked'}
    manifest['training_task_families']=len({r['group'] for r in train if r['kind']=='task'})
    manifest['training_general_types']=len({r['general_type'] for r in train if r['kind']=='general'})
    manifest['retired_training_records_by_reason']=dict(Counter(r['reason'] for r in retired))
    manifest['scenario_count_definition']='Task counts are fault/task families. General counts are question types; numeric instances are not counted as separate task scenes.'
    write_json(DATA/'manifest.json',manifest);write_json(DATA/'source-review-lock.json',{'sha256':sha256(DATA/'source-review.jsonl'),'review_status':'agent reviewed; independent human expert review not claimed'})
    write_json(DATA/'curation/active-seal.json',{'manifest_sha256':sha256(DATA/'manifest.json'),
        'test_files':{s+'.jsonl':sha256(DATA/(s+'.jsonl')) for s in ('test','general-test')},'model_evaluation_run':False})
    write_json(REPORT/'inventory.json',manifest)
    write_json(REPORT/'build-check.json',{'passed':True,'prefix_variants_retired':len(retired),'duplicate_ids':0,'duplicate_full_contexts':0,
        'scenario_overlap':manifest['task_split_group_overlap'],'old_acceptance_bytes_preserved':all(sha256(archived/name)==provenance['files'][name] for name in ('test.jsonl','general-test.jsonl'))})
    print(json.dumps({'counts':counts,'scenarios':scenes,'training_references':manifest['unique_training_references'],
        'multi_turn_training':manifest['multi_turn_training_records'],'retired_training_records':len(retired)},ensure_ascii=False))
    return manifest

if __name__=='__main__':build()
