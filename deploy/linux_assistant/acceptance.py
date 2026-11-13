"""Run on RK3576: raw development/final answers or measured performance/stability.

No model-quality pass is inferred from successful API calls. Final evaluation
requires a lock whose question and model hashes match the actual files.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import time
import tempfile
from frontend import NativeTokenizer,Runner,bounded_history,verify_bundle


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8<<20):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def question_context(messages):
    """Same target exclusion as PC evaluation; keep earlier assistant history."""
    context=list(messages)
    if context and context[-1]['role']=='assistant':context=context[:-1]
    if not context or context[-1]['role']!='user':raise ValueError('Evaluation context must end in a user turn')
    return context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['dev','final','performance','stability','boundary'],required=True)
    p.add_argument('--questions',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--final-lock',type=Path);a=p.parse_args()
    manifest=verify_bundle(a.root);tok=NativeTokenizer(a.root/'tokenizer')
    if a.output.exists():raise FileExistsError('A measured run is immutable; inspect existing evidence before retry')
    questions=[]
    if a.questions:questions=[json.loads(s) for s in a.questions.read_text(encoding='utf8').splitlines() if s.strip()]
    if a.mode in ('dev','final') and not questions:raise ValueError('Question file required')
    if a.mode=='final':
        if not a.final_lock:raise ValueError('Final lock required')
        lock=json.loads(a.final_lock.read_text(encoding='utf8'))
        if lock['questions_sha256']!=digest(a.questions) or lock['model_sha256']!=digest(a.root/'model.rkllm'):
            raise ValueError('Final model/questions do not match lock')
        marker=a.root/'final-round.started.json'
        with marker.open('x',encoding='utf8') as f:json.dump(lock,f)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    runner=Runner(a.root);rows=[];started=time.monotonic();failure=None
    def run(messages,identity,extra=None):
        preprocessing_started=time.monotonic()
        context=question_context(messages)
        prompt,ids=tok.encode([{'role':'system','content':manifest['system_prompt']},*context])
        if len(ids)>1792:raise ValueError('Question exceeds frozen deployment input limit')
        preprocessing_seconds=time.monotonic()-preprocessing_started
        result=runner.generate(ids)
        result['end_to_end_seconds']+=preprocessing_seconds
        if result['end_to_end_first_text_seconds'] is not None:result['end_to_end_first_text_seconds']+=preprocessing_seconds
        row={'id':identity,'messages':messages,'input_messages':context,'input_ids':ids,'prompt':prompt,**result,**(extra or {}),
             'model_sha256':manifest['files']['model.rkllm']}
        if a.mode=='final':row['final_lock_sha256']=digest(a.final_lock)
        rows.append(row)
        with a.output.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return result
    try:
        # Warmup excluded from all reported request counts and percentile statistics.
        _,ids=tok.encode([{'role':'system','content':manifest['system_prompt']},{'role':'user','content':'你好'}])
        runner.generate(ids)
        if a.mode in ('dev','final'):
            for q in questions:run(q['messages'],q['id'],{'kind':q['kind'],'category':q['category'],'group':q['group'],
                  'reference':q['reference'],'rubric':q['rubric'],'accepted':None,'semantic_review':'pending',
                  **({'grading':q['grading']} if 'grading' in q else {})})
        elif a.mode=='boundary':
            checks={}
            for name,question in [('empty','  '),('overlong','很长的问题'*4096)]:
                try:bounded_history(tok,manifest['system_prompt'],[],question)
                except ValueError:checks[name]=True
                else:checks[name]=False
            history=[{'role':'user','content':'旧问题'*1500},{'role':'assistant','content':'旧答案'*1500}]
            retained,prompt,ids=bounded_history(tok,manifest['system_prompt'],history,'当前问题')
            checks['trim_complete_turn']=not retained and len(history)==2 and '当前问题' in prompt and manifest['system_prompt'] in prompt and len(ids)<=1792
            try:unexpected=Runner(a.root,model='intentionally-absent-model.rkllm')
            except RuntimeError:checks['wrong_model_path']=True
            else:unexpected.close();checks['wrong_model_path']=False
            with tempfile.TemporaryDirectory(dir=a.root) as directory:
                root=Path(directory);(root/'sample').write_bytes(b'changed')
                save(root/'bundle-manifest.json',{'files':{'sample':hashlib.sha256(b'original').hexdigest()}})
                try:verify_bundle(root)
                except ValueError:checks['digest_mismatch']=True
                else:checks['digest_mismatch']=False
            _,ids=tok.encode([{'role':'system','content':manifest['system_prompt']},{'role':'user','content':'请连续列举并解释二十条Linux命令。'}])
            sent=[False]
            def stop_after_first_text(text):
                if text and not sent[0]:sent[0]=True;runner.interrupt()
            result=runner.generate(ids,stop_after_first_text)
            checks['interrupt']=sent[0] and result.get('interrupted') is True
            run([{'role':'user','content':'只回答：恢复成功'}],'after-interrupt')
            checks['request_after_interrupt']=True
            summary_checks=checks
            if not all(checks.values()):raise RuntimeError('Boundary checks failed: '+json.dumps(checks))
        elif a.mode=='performance':
            prompts=['简述pwd命令的作用。','简述uname -m命令的作用。','简述whoami命令的作用。','简述内存与存储的区别。','解释SHA256校验。',
                     '什么是工作目录？','什么是绝对路径？','什么是运行库？','什么是日志？','什么是进程？',
                     '查看文件前几行用什么命令？','解释文件权限。','怎样查看磁盘空间？','什么是分词？','什么是模型量化？',
                     '什么是上下文窗口？','怎样查看Python路径？','什么是板卡架构？','什么是流式输出？','为什么记录软件版本？']
            for i,q in enumerate(prompts):
                _,ids=tok.encode([{'role':'system','content':manifest['system_prompt']},{'role':'user','content':q}])
                if len(ids)>128:raise ValueError('Performance prompt exceeds 128 tokens')
                run([{'role':'user','content':q}],f'perf-{i:02d}')
        else:
            # Each new session has a unique marker; every query resubmits only that session's history.
            for session in range(10):
                marker=f'CASE-{session:03d}';history=[]
                for turn in range(5):
                    q=f'我的项目代号是{marker}，请只回复这个代号。' if turn==0 else '请只输出我前面给出的项目代号。'
                    retained,_,_=bounded_history(tok,manifest['system_prompt'],history,q)
                    result=run([*retained,{'role':'user','content':q}],f'session-{session}-{turn}',
                               {'session':session,'turn':turn,'expected_marker':marker})
                    if marker not in result['answer'] or any(f'CASE-{j:03d}' in result['answer'] for j in range(10) if j!=session):
                        raise RuntimeError('Session recall/isolation check failed')
                    history=[*retained,{'role':'user','content':q},{'role':'assistant','content':result['answer']}]
            n=0
            while time.monotonic()-started<3600 or len(rows)<100:
                q=['简要说明内存和存储的区别。','怎样查看当前工作目录？','怎样确认模型传输完整？','排查运行库缺失应收集什么？'][n%4]
                run([{'role':'user','content':q}],f'stability-{n:04d}');n+=1
    except BaseException as exc:
        failure=str(exc);raise
    finally:
        runner.close();elapsed=time.monotonic()-started
        summary={'completed':failure is None,'mode':a.mode,'requests':len(rows),'seconds':elapsed,
                 'error':failure,'model_sha256':manifest['files']['model.rkllm'],
                 'semantic_quality_pass':None,'combined_peak_rss_kib':runner.peak_combined_kib}
        if a.mode=='performance' and rows:
            values=sorted(r['end_to_end_first_text_seconds'] for r in rows if r['end_to_end_first_text_seconds'] is not None)
            rates=[r['decode_tokens_per_second_wall'] for r in rows if r['decode_tokens_per_second_wall']>0]
            summary.update(first_text_p95_seconds=values[math.ceil(.95*len(values))-1] if values else None,
                           decode_median_tokens_per_second=statistics.median(rates) if rates else None,
                           decode_metric_valid_samples=len(rates),
                           decode_metric='(nonterminal token count - 1) / first-to-last callback wall interval')
            summary['passed']=failure is None and len(rows)==20 and len(values)==20 and len(rates)>=18 and summary['first_text_p95_seconds']<=3 and summary['decode_median_tokens_per_second']>=4 and runner.peak_combined_kib<=3*1024**2
        elif a.mode=='stability':
            summary['passed']=failure is None and elapsed>=3600 and len(rows)>=100 and runner.peak_combined_kib<=3*1024**2
        elif a.mode=='boundary':
            summary['checks']=locals().get('summary_checks',locals().get('checks',{}))
            summary['passed']=failure is None and len(summary['checks'])==7 and all(summary['checks'].values())
        save(a.output.with_suffix('.summary.json'),summary)


if __name__=='__main__':main()
