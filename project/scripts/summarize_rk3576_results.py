"""Validate two complete board rounds and summarize raw runtime observations."""
import json
import statistics
from scripts.common import WORKSPACE,write_json

def main():
    report=WORKSPACE/'reports/rk3576-qwen-20261018';board=report/'board'
    summaries=[];md=['# RK3576 两模型原始问答','',
        '每模型预热一题后，独立执行八题两轮。使用PC原生输入Token、top_k=1、重复惩罚1、最多256个新Token。原始JSONL保留返回码、回调状态与性能字段。','']
    for name in ['Qwen3.5-2B','Qwen3-4B-Instruct-2507']:
        path=board/f'{name}.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        questions=json.loads((WORKSPACE/'exports/rk3576'/name/'questions.json').read_text(encoding='utf8'))
        expected={q['id'] for q in questions};lookup={q['id']:q['question'] for q in questions}
        complete=(len(rows)==17 and sum(r['warmup'] for r in rows)==1 and
                  all(r['return_code']==0 and not r['callback_error'] and r['finished'] and r['output'].strip() for r in rows) and
                  all({r['question_id'] for r in rows if r['round']==round}==expected and sum(r['round']==round for r in rows)==8 for round in [1,2]))
        measured=[r for r in rows if not r['warmup']]
        rates=[r['generate_tokens']*1000/r['generate_ms'] for r in measured if r['generate_ms']>0]
        metrics={'model':name,'completed':complete,'measured_questions':len(measured),
                 'median_first_token_seconds':statistics.median(r['first_token_seconds'] for r in measured) if measured else None,
                 'median_runtime_generate_tokens_per_second':statistics.median(rates) if rates else None,
                 'max_process_hwm_mib':max(r['hwm_kib']/1024 for r in rows) if rows else None,
                 'runtime_memory_mb_max':max(r['runtime_memory_mb'] for r in rows) if rows else None,
                 'round_end_rss_mib':{str(i):next((r['rss_kib']/1024 for r in reversed(rows) if r['round']==i),None) for i in [0,1,2]},
                 'at_length_limit':sum(r['stop_reason']=='runtime_finish_at_limit' for r in measured)}
        summaries.append(metrics);md.extend([f'## {name}',''])
        for round in [1,2]:
            md.extend([f'### 第{round}轮',''])
            for row in [r for r in rows if r['round']==round]:
                md.extend([f"**{lookup[row['question_id']]}**",'','```text',row['output'],'```','',
                    f"停止：{row['stop_reason']}；首Token回调：{row['first_token_seconds']:.3f}s；总耗时：{row['seconds']:.3f}s。",''])
    write_json(report/'board-summary.json',{'completed':all(s['completed'] for s in summaries),'models':summaries,
        'limits':'Short prompts only, no frequency locking; runtime-defined generation throughput; two rounds do not establish long-term memory stability.'})
    (report/'BOARD_ANSWERS.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    if not all(s['completed'] for s in summaries):raise RuntimeError('Board validation incomplete; inspect raw logs')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
