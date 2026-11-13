"""Unattended reviewed-data Qwen3.5-2B smoke, resume, LoRA and development evaluation."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.common import WORKSPACE,sha256,write_json,utc
from scripts.linux_assistant.common import config,jsonl,training_rows,fingerprint,verify_data
from scripts.linux_assistant.review_training_data import validate_training_review
from scripts.linux_assistant.baseline import exclusive_job,prepare

def verify_semantic_baseline(root,raw):
    """Generation completion alone must not release a new-data training run."""
    from scripts.linux_assistant.common import reviewed_scores
    path=Path(root)/'answers.reviewed.jsonl'
    if not path.exists():raise ValueError('New baseline requires complete semantic review before retraining')
    reviewed=jsonl(path)
    if len(reviewed)!=len(raw):raise ValueError('Baseline semantic review coverage differs')
    for actual,decision in zip(raw,reviewed):
        for name in ('id','messages','reference','rubric','grading','answer','prompt','input_ids','generated_ids'):
            if actual.get(name)!=decision.get(name):raise ValueError('Baseline semantic review changed actual output: '+name)
    return {'reviewed_answers_sha256':sha256(path),'scores':reviewed_scores(reviewed)}

def verify_baseline(c,tokenizer):
    root=WORKSPACE/c['baseline_report']
    identity=json.loads((root/'identity.json').read_text('utf8'))
    summary=json.loads((root/'summary.json').read_text('utf8'))
    rows=jsonl(c['data']/'dev.jsonl')+jsonl(c['data']/'general-dev.jsonl')
    if not summary.get('completed') or summary.get('answers_sha256')!=sha256(root/'answers.jsonl'):
        raise ValueError('Clean baseline is incomplete or its output changed')
    expected={'revision':c['revision'],'rows':fingerprint(rows),'system_prompt':c['system_prompt'],
        'max_new_tokens':c['max_new_tokens'],'context_length':c['context_length'],
        'greedy':True,'thinking':False,'prompt_policy':'context_through_last_user_exclude_target_assistant'}
    for key,value in expected.items():
        if identity.get(key)!=value:raise ValueError('Baseline policy differs: '+key)
    prepared=prepare(c,tokenizer,rows);answers=jsonl(root/'answers.jsonl')
    if len(answers)!=len(prepared):raise ValueError('Baseline coverage differs')
    for p,r in zip(prepared,answers):
        if r['id']!=p['row']['id'] or r['prompt']!=p['prompt'] or r['input_ids']!=p['input_ids']:
            raise ValueError('Baseline actual input differs: '+r['id'])
    semantic=verify_semantic_baseline(root,answers) if c.get('require_baseline_semantic_review') else None
    return {'directory':str(root),'answers_sha256':sha256(root/'answers.jsonl'),
        'samples':len(answers),'exact_input_token_parity':True,
        'semantic_review':semantic or 'pending','reused_clean_baseline':True}

def expected_update_count(c):
    """Match train.execute(): gradient_accumulation is examples per optimizer update."""
    rows=training_rows(c)
    accumulation=c['training']['gradient_accumulation']
    epochs=c['training']['epochs']
    if accumulation<1 or epochs<1:
        raise ValueError('Training epochs and effective examples per update must be positive')
    return math.ceil(len(rows)*epochs/accumulation)

def preflight(c,config_path=None,tokenizer=None):
    """Run all data and baseline release gates without allocating a model or CUDA context."""
    if 'baseline' in c:
        from scripts.linux_assistant.baseline import validate_batch_candidates
        validate_batch_candidates(c)
    if c.get('experiment_scope')=='controlled_1000_skill_rebuild':
        from scripts.linux_assistant.train import checkpoint_policy
        checkpoint_policy(c['training'],expected_update_count(c))
        checkpoint_policy(c['training'],4,smoke=True)
    review=validate_training_review(c)
    verify_data(c)
    cleanup_path=WORKSPACE/c['cleanup_record']
    cleanup=json.loads(cleanup_path.read_text(encoding='utf-8-sig'))
    if cleanup.get('completed') is not True or any(Path(p).exists() for p in cleanup['deleted_targets']):
        raise ValueError('Previous experiment cleanup not completed')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    baseline=verify_baseline(c,tokenizer)
    from scripts.linux_assistant.modeling import training_data
    training_records,encoded=training_data(c,tokenizer)
    if len(training_records)!=len(encoded) or not encoded:
        raise ValueError('Preflight did not encode every training record')
    lengths=[len(row['input_ids']) for row in encoded]
    max_index=max(range(len(lengths)),key=lengths.__getitem__)
    training_encoding={
        'records':len(training_records),
        'input_tokens':sum(lengths),
        'supervised_tokens':sum(sum(token!=-100 for token in row['labels']) for row in encoded),
        'max_sequence_length':max(lengths),
        'max_sequence_record_id':training_records[max_index]['id'],
        'configured_max_length':c['max_length'],
        'all_rows_fit_without_truncation':True,
        'gpu_fit_proven':False,
    }
    release_contract=None
    if c.get('experiment_scope')=='controlled_1000_skill_rebuild':
        from scripts.linux_assistant.release_contract import validate_release_contract
        release_contract=validate_release_contract(c,training_records,encoded)
    return {'passed':True,'data_review':review,'baseline':baseline,
        'cleanup_record_sha256':sha256(cleanup_path),
        'configuration_sha256':sha256(config_path) if config_path else None,
        'training_encoding':training_encoding,'release_contract':release_contract,
        'test_split_loaded':False,
        'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
            ('retrain.py','train.py','evaluate.py','modeling.py','review_training_data.py')}}

def record_preflight(c,config_path,tokenizer=None):
    """Persist failed launch gates too, without replacing earlier run results."""
    attempt=c['report']/'preflight-attempts'/(
        utc().replace(':','-')+f'-{os.getpid()}.json')
    identity={'configuration_sha256':sha256(config_path),'utc':utc(),
              'training_started':False,'model_loaded':False,
              'final_acceptance_generated':False}
    try:
        result=preflight(c,config_path,tokenizer)
    except Exception as error:
        failure={**identity,'passed':False,'error':str(error),'error_type':type(error).__name__}
        if c.get('experiment_scope')=='controlled_1000_skill_rebuild' and Path(c.get('data','')).is_dir():
            try:
                from scripts.linux_assistant.preparation_audit import audit
                diagnostic=attempt.with_name(attempt.stem+'-all-gates.json')
                all_gates=audit(c,Path(c['data']),diagnostic)
                failure.update(all_gate_diagnostics=str(diagnostic),issue_count=len(all_gates['issues']))
            except Exception as diagnostic_error:
                failure['diagnostic_error']=str(diagnostic_error)
        write_json(attempt,failure)
        raise
    write_json(attempt,{**identity,**result})
    return result

def comparison(c,baseline):
    from scripts.linux_assistant.common import screen
    base=jsonl(Path(baseline['directory'])/'answers.jsonl')
    final_step=expected_update_count(c)
    candidate_path=c['runs']/f'lr-1e-4/dev-{final_step}.jsonl'
    candidate=jsonl(candidate_path)
    if len(base)!=len(candidate):raise ValueError('Final development evaluation incomplete')
    pairs=[]
    for left,right in zip(base,candidate):
        if left['id']!=right['id'] or left['prompt']!=right['prompt'] or left['input_ids']!=right['input_ids']:
            raise ValueError('Comparison input identity mismatch: '+right['id'])
        pairs.append({'id':right['id'],'kind':right['kind'],'category':right['category'],
            'messages':right['messages'],**({'grading':right['grading']} if 'grading' in right else {}),
            'baseline_answer':left['answer'],'adapter_answer':right['answer'],'reference':right['reference'],
            'baseline_stop':left['stop_reason'],'adapter_stop':right['stop_reason'],
            'baseline_screen':screen(left,left['answer'])['screen_pass'],
            'adapter_screen':screen(right,right['answer'])['screen_pass'],
            'accepted':None,'semantic_review':'pending'})
    from scripts.linux_assistant.baseline import atomic_rows
    atomic_rows(c['report']/'comparison-review-queue.jsonl',pairs)
    screens={kind:{'samples':sum(r['kind']==kind for r in pairs),
        'baseline_necessary_conditions':None if any('grading' in r for r in base) else sum(r['kind']==kind and r['baseline_screen'] for r in pairs),
        'adapter_necessary_conditions':None if any('grading' in r for r in candidate) else sum(r['kind']==kind and r['adapter_screen'] for r in pairs)} for kind in ('task','general')}
    result={'completed':True,'final_step':final_step,'candidate':str(candidate_path.parent/f'checkpoint-{final_step}'),
        'input_token_parity':True,'development_samples':len(pairs),'necessary_condition_screens':screens,
        'semantic_review':'pending','accepted_score':None,'checkpoint_selection':'pending semantic review; no regex-based winner',
        'final_acceptance_run':False,'second_learning_rate_run':False,'merged_or_quantized':False,
        'candidate_answers_sha256':sha256(candidate_path),'baseline':baseline}
    write_json(c['report']/'development-comparison.json',result)
    return result

def run(c,config_path):
    from scripts.linux_assistant.train import execute,reload_smoke
    report=c['report'];report.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();began=utc();stage='preflight'
    def state(value,**more):
        write_json(report/'status.json',{'state':value,'stage':stage,'pid':os.getpid(),
            'started_utc':began,'updated_utc':utc(),'elapsed_seconds':time.monotonic()-started,**more})
    with exclusive_job(report/'worker.lock'),exclusive_job(WORKSPACE/'runs/linux-assistant-gpu.lock'):
        if (report/'status.json').exists():raise FileExistsError('Run already has a status; use a new experiment path')
        state('running')
        try:
            checked=preflight(c,config_path)
            review=checked['data_review']
            baseline=checked['baseline']
            write_json(report/'preflight.json',checked)
            (report/'dependencies-lock.txt').write_text(subprocess.check_output(
                [sys.executable,'-m','pip','freeze'],text=True,encoding='utf8'),encoding='utf8')
            (report/'configuration.json').write_bytes(config_path.read_bytes())
            stage='smoke_and_resume';state('running',approved_training_records=review['approved_records'])
            smoke_run=c['runs']/'smoke'
            execute(c,smoke_run,smoke=True,stop_after=2)
            smoke=execute(c,smoke_run,smoke=True,resume=smoke_run/'checkpoint-2')
            reload_smoke(c,smoke_run)
            smoke=json.loads((smoke_run/'progress.json').read_text('utf8'))
            if not all(smoke.get(k) for k in ('completed','adapter_changed','frozen_base_unchanged','reload_passed')):
                raise ValueError('Smoke update/freeze/resume/reload did not pass')
            # Estimate the two expensive phases from this run's four-update smoke
            # and the exact-data baseline generation.  Keep them separate because
            # output length can dominate evaluation while sequence length dominates
            # training.
            expected_steps=expected_update_count(c)
            smoke_elapsed=smoke['history'][-1]['elapsed_seconds']
            training_estimate=smoke_elapsed/smoke['history'][-1]['step']*expected_steps
            baseline_summary=json.loads((Path(baseline['directory'])/'summary.json').read_text('utf8'))
            evaluation_seconds=baseline_summary.get('evaluation_session_seconds',baseline_summary.get('session_seconds'))
            checkpoint_count=len(c['training'].get('checkpoint_steps',[])) or math.ceil(expected_steps/c['training']['save_steps'])
            estimate={'basis':'same-data four-update smoke and full baseline generation',
                'training_updates':expected_steps,'checkpoint_evaluations':checkpoint_count,
                'estimated_training_seconds':training_estimate,
                'estimated_development_generation_seconds':evaluation_seconds*checkpoint_count,
                'estimated_total_seconds_excluding_semantic_review':training_estimate+evaluation_seconds*checkpoint_count,
                'not_an_eta':'Model loading, checkpoint saving and answer-length variation are not fully represented.'}
            write_json(report/'time-estimate.json',estimate)
            stage='formal_training_and_development';state('running',smoke_passed=True,time_estimate=estimate)
            execute(c,c['runs']/'lr-1e-4')
            stage='development_comparison';state('running')
            result=comparison(c,baseline)
            stage='awaiting_semantic_review';state('awaiting_semantic_review',
                training_completed=True,development_generation_completed=True,
                development_samples=result['development_samples'],
                semantic_review='pending',quality_gate='not yet assessed',final_acceptance_run=False)
            text=f'''# Qwen3.5-2B复核后重新微调与开发评测

训练、保存恢复检查及最终{result['development_samples']}题开发生成已完成。总耗时{(time.monotonic()-started)/60:.1f}分钟。

训练仅使用通过逐条独立内容复核的{review['approved_records']}条数据，使用BF16 LoRA、micro-batch {c['training']['micro_batch']}、有效批次{c['training']['gradient_accumulation']}、学习率{c['training']['learning_rate']}和{c['training']['epochs']}个epoch。检查点策略为{c['training'].get('checkpoint_steps', c['training'].get('save_steps'))}；完整开发集检查为{c['training'].get('full_development_at_each_checkpoint', False)}。只有同一完整开发集、同一预算和同一评分协议的结果才能用于选型。

原模型使用相同数据与提示的{result['development_samples']}题基线，最终逐题核对提示文本和Token ID完全一致。比较记录见development-comparison.json；逐题回答见comparison-review-queue.jsonl。

必要条件筛查结果：{json.dumps(result['necessary_condition_screens'],ensure_ascii=False)}。

这些是规则筛查，不是语义通过率。语义评审和检查点选择尚待完成，未宣布LoRA提升或工程门槛通过。未自动运行第二学习率、合并、量化或封存验收题。

数据复核为代理逐条复核，不等同独立人工专家审阅或Linux/板端执行。状态和所有错误会自动保存，无需实时监控；流程结束后可按status.json检查结果。
'''
            (report/'REPORT.md').write_text(text,encoding='utf8')
            print('Completed: '+str(report/'REPORT.md'),flush=True)
        except BaseException as error:
            write_json(report/'failure.json',{'stage':stage,'error':str(error),'type':type(error).__name__,
                'traceback':traceback.format_exc(),'utc':utc(),'action':'Stopped; do not switch models or scale automatically.'})
            state('failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true',help='Run data, cleanup and baseline gates without starting model training')
    a=p.parse_args();c=config(a.config);resolved=a.config.resolve()
    if a.preflight_only:
        report=c['report'];report.mkdir(parents=True,exist_ok=True)
        with exclusive_job(report/'worker.lock'),exclusive_job(WORKSPACE/'runs/linux-assistant-gpu.lock'):
            if (report/'status.json').exists():raise FileExistsError('Run already has a status; use a new experiment path')
            result=record_preflight(c,resolved)
        print(json.dumps({'preflight':'passed',**result},ensure_ascii=False,default=str),flush=True)
    else:run(c,resolved)
