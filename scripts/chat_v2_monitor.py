"""读取持久化控制器、日志与真实 GPU 状态，可同步写入课程进度页。"""
from scripts.common import RUNS, ROOT, read_json, write_json, utc
import argparse
import json
import subprocess
import psutil
from pathlib import Path

def safe_read(path):
    try:
        return read_json(path) if path.exists() else None
    except (OSError,json.JSONDecodeError):
        return None

def monitor(update_docs=False):
    state=safe_read(RUNS/'reports/pipeline_status.json') or {}
    prep=safe_read(RUNS/'reports/preparation_progress.json')
    ledger=safe_read(RUNS/'reports/gpu_budget.json') or []
    command=state.get('command',[])
    base_folder=RUNS/'checkpoints/50m-base-v2'
    if 'training.train' in command and '--stage' not in command and '--output' in command:
        base_folder=Path(command[command.index('--output')+1])
    elif 'training.train' in command and '--stage' in command and command[command.index('--stage')+1]=='pretrain':
        base_folder=Path(command[command.index('--output')+1])
    elif state.get('base_checkpoint'):
        base_folder=Path(state['base_checkpoint']).parent
    metrics=base_folder/'metrics.jsonl'
    last=None
    initial=safe_read(base_folder/'validation_latest.json')
    validation=dict(step=initial['step'],loss=initial['validation_loss']) if initial else None
    if metrics.exists():
        with metrics.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'input_tokens' in row and 'loss' in row:
                    last=row
                if 'validation_loss' in row and (validation is None or row['step']>=validation['step']):
                    validation=dict(step=row['step'],loss=row['validation_loss'])
    pids={key:bool(state.get(key) and psutil.pid_exists(state[key])) for key in ('pid','child_pid')}
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
    result=dict(updated_utc=utc(),pipeline=state,process_alive=pids,preparation=prep,last_pretrain_metric=last,last_periodic_validation=validation,gpu_minutes=sum(r['seconds'] for r in ledger)/60,gpu=gpu,capability_status=state.get('capability_status','not_accepted'))
    result['pretrain_folder']=str(base_folder)
    result['waiting_for_user_review']=state.get('phase')=='awaiting_development_review' and not any(pids.values())
    result['actual_processes']={}
    for key in ('pid','child_pid'):
        try:
            if state.get(key):
                process=psutil.Process(state[key])
                result['actual_processes'][key]=dict(pid=process.pid,created=process.create_time(),command=process.cmdline())
        except psutil.Error:pass
    # 阶段已进入SFT时，单列当前Chat数据/进度，不能拿Base loss冒充SFT loss。
    sft_folder=None
    if 'training.train' in command and '--stage' in command and command[command.index('--stage')+1]=='sft':
        sft_folder=Path(command[command.index('--output')+1])
    checkpoint=state.get('checkpoint')
    if not checkpoint and state.get('answers'):
        paths=state['answers'] if isinstance(state['answers'],list) else [state['answers']]
        answers=safe_read(Path(paths[-1]))
        checkpoint=answers.get('checkpoint') if answers else None
    if checkpoint:
        folder=Path(checkpoint).parent
        config=safe_read(folder/'run_config.json')
        if config and config['settings']['stage']=='sft':
            sft_folder=folder
    if sft_folder:
        sft_config=safe_read(sft_folder/'run_config.json') or {}
        sft_last=None
        metric_path=sft_folder/'metrics.jsonl'
        if metric_path.exists():
            for line in metric_path.read_text(encoding='utf-8').splitlines():
                try:sft_last=json.loads(line)
                except json.JSONDecodeError:continue
        sft_summary=safe_read(sft_folder/'summary.json')
        result['current_sft']=dict(folder=str(sft_folder),dataset=sft_config.get('dataset'),
                                   last_metric=sft_last,validation=safe_read(sft_folder/'validation_latest.json'),
                                   summary={k:v for k,v in sft_summary.items() if k!='history'} if sft_summary else None,
                                   input_token_scope='batch positions including padding; supervised_tokens count targets only')
    result['sft_gpu_minutes']=sum(r['seconds'] for r in ledger if r['stage'].startswith('sft-'))/60
    performance=safe_read(RUNS/'reports/execution_profile.json')
    if performance:
        result['execution_profile']=performance
    write_json(RUNS/'reports/monitor_latest.json',result)
    if update_docs:
        text=['# 50M 续训进度','',f"更新时间（UTC）：{result['updated_utc']}。本页读取实际进程与日志，不代表聊天能力已经达标。",'',f"当前阶段：`{state.get('phase','unknown')}`；控制器存活：{pids['pid']}；最近子进程存活：{pids['child_pid']}。",'',f"新实验累计 GPU：{result['gpu_minutes']:.2f} / 1440 分钟。",'',f"能力状态：`{result['capability_status']}`。详细验收见 [训练闭环说明](50M聊天训练闭环.md)。"]
        if result['waiting_for_user_review']:
            text+=['','当前没有GPU工作，已释放控制器、锁和临时保活；等待用户请求审阅，不是运行停滞。']
        if prep:
            text+=['',f"数据准备最新记录：`{json.dumps(prep,ensure_ascii=False)}`。"]
        if last:
            text+=['',f"最近预训练更新：{last['step']:,} 步；本阶段输入 {last['input_tokens']:,} Token；训练 loss {last['loss']:.4f}；速度 {last['tokens_per_second']:.1f} Token/s。"]
        if validation:
            text+=['',f"最近可用验证：第 {validation['step']:,} 步，loss {validation['loss']:.4f}。验证设置与旧实验不同，不能直接当作同一条曲线比较。"]
        if result.get('current_sft'):
            sft=result['current_sft']
            metric=sft['summary'] or sft['last_metric']
            text+=['',f"当前 SFT：`{sft['folder']}`；数据清单：`{json.dumps(sft['dataset'],ensure_ascii=False)}`。"]
            if metric:
                text+=['',f"当前候选本阶段已训练 {metric['step']:,} 步，监督 {metric['supervised_tokens']:,} Token；输入位置 {metric['input_tokens']:,}（含padding，不能当作有效输入吞吐）。"]
            if sft['last_metric'] and 'effective_tokens_per_second' in sft['last_metric']:
                row=sft['last_metric']
                text+=['',f"本次运行含验证和保存的非填充输入吞吐 {row['effective_tokens_per_second']:.1f} Token/s；监督吞吐 {row['supervised_tokens_per_second']:.1f} Token/s。计数范围：{row.get('effective_input_token_scope')}。"]
            if sft['validation']:
                text+=['',f"当前 SFT 验证 loss：{sft['validation']['validation_loss']:.4f}。不同数据版本的验证集不同，不能与此前 SFT loss 直接画同一条曲线。"]
        text+=['',f"SFT与probe累计 GPU：{result['sft_gpu_minutes']:.2f} / 180 分钟。"]
        if performance:
            text+=['',f"实际预训练执行设置：`{json.dumps(performance['pretrain'],ensure_ascii=False)}`。对照短测速度比 {performance['speedup']:.2f}；固定验证微批次保持2。",'',f"性能回退已发生：{performance.get('rollback_used',False)}。详细数值和切换损失见 [效率实测](Win11训练效率优化.md)。"]
        text+=['','```text',gpu,'```','',r'原始日志和检查点：`D:\LubanLM-Runs\chat-v2`。未通过独立验收前，训练闭环保持未完成。','']
        (ROOT/'docs/50M续训进度.md').write_text('\n'.join(text),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--update-docs',action='store_true')
    args=p.parse_args()
    print(json.dumps(monitor(args.update_docs),ensure_ascii=False,indent=2))
