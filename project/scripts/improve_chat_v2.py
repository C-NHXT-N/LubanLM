"""预算内独立 SFT 改进候选；精确恢复、逐题开发审阅、随后独立验收。"""
from scripts.common import DATA, RUNS, ROOT, read_json, write_json, sha256, utc
from scripts.train_chat_v2 import used, status, run, wait_review, profile_options, DEFAULT_PROFILE, ReviewPending
from pathlib import Path
import os
import argparse
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=RUNS/'reports/sft_improvement_plan.json')
    plan_path=parser.parse_args().plan
    plan = read_json(plan_path)
    suite_path=Path(plan['suite_file']) if plan.get('suite_file') else None
    if suite_path and sha256(suite_path)!=plan['suite_sha256']:
        raise ValueError('frozen suite changed')
    suite_options=['--suite-file',suite_path] if suite_path else []
    folder = RUNS/'checkpoints'/plan['name']
    init = Path(plan['init_checkpoint'])
    if sha256(init) != plan['init_sha256']:
        raise ValueError('immutable initialization checkpoint changed')
    data = DATA/'processed'/plan['data_version']
    manifest = read_json(data/'manifest.json')
    for kind in ('train', 'validation'):
        if sha256(data/f'{kind}-tokens.jsonl') != manifest[f'{kind}_sha256']:
            raise ValueError('versioned SFT dataset changed')
    owner = RUNS/'pipeline.lock'
    with owner.open('x') as stream:
        stream.write(str(os.getpid()))
    awake=subprocess.Popen([sys.executable,'-m','scripts.keep_awake','--pid',str(os.getpid())],cwd=ROOT,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        status('starting_sft_improvement', candidate=plan['name'], plan=str(plan_path))
        summary_path = folder/'summary.json'
        if not summary_path.exists():
            remaining = min(180-used('sft-'), 1380-used())
            if remaining < 5:
                status('not_accepted', reason='SFT or total budget exhausted', checkpoint=str(init))
                return
            settings_path = folder/'run_config.json'
            if settings_path.exists():
                settings = read_json(settings_path)['settings']
                # 原目标与设置不变；minutes仅控制本次时间，取剩余原预算。
                options = []
                for key in ('config', 'stage', 'output', 'lr', 'epochs', 'warmup_steps',
                            'micro_batch', 'accumulation', 'execution', 'attention_backend',
                            'optimizer_backend', 'validation_micro_batch', 'sft_train',
                            'sft_validation', 'save_every_minutes', 'log_every', 'steps',
                            'sft_batching','sft_sampling','sft_objective','prefetch_batches',
                            'validation_every_steps','early_stop_patience','parent_sft_tokens'):
                    if settings.get(key) is not None:
                        options += ['--'+key.replace('_', '-'), settings[key]]
                for key in ('full_sft_validation', 'pin_memory','source_validation'):
                    if settings.get(key):
                        options.append('--'+key.replace('_', '-'))
                checkpoints = [p for p in [folder/'latest.pt', folder/'last.pt'] if p.exists()]
                if not checkpoints:
                    raise RuntimeError('incomplete attempt without recoverable checkpoint; retain files for audit')
                options += ['--resume', max(checkpoints, key=lambda p:p.stat().st_mtime),
                            '--minutes', min(plan['minutes'], remaining)]
            else:
                profile_file = RUNS/'reports/execution_profile.json'
                profile = read_json(profile_file).get('sft', DEFAULT_PROFILE) if profile_file.exists() else DEFAULT_PROFILE
                options = ['--config', ROOT/'configs/config_50m.json', '--stage', 'sft', '--init', init,
                           '--output', folder, '--lr', plan['lr'], '--epochs', plan['epochs'],
                           '--minutes', min(plan['minutes'], remaining), '--warmup-steps', 100,
                           '--full-sft-validation', '--save-every-minutes', 10, '--log-every', 100,
                           '--sft-train', data/'train-tokens.jsonl',
                           '--sft-validation', data/'validation-tokens.jsonl']+profile_options(profile)
                for key in ('steps','sft_batching','sft_sampling','sft_objective','prefetch_batches','validation_every_steps','early_stop_patience','parent_sft_tokens'):
                    if key in plan:options+=['--'+key.replace('_','-'),plan[key]]
                if plan.get('source_validation'):options+=['--source-validation']
            run('training.train', options, 'sft-'+plan['name'])
        checkpoint = folder/plan.get('checkpoint_name','last.pt')
        raw = RUNS/'reports'/(plan['name']+'-dev.json')
        if not raw.exists():
            run('eval.chat_suite', ['--checkpoint', checkpoint, '--split', 'dev', '--output', raw,
                                   '--minutes', min(15, 1440-used())]+suite_options, plan['name']+'-dev')
        if read_json(raw)['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('development answers refer to a different checkpoint')
        review = wait_review(raw)
        record = dict(plan=plan, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      review=str(raw.with_suffix('.review.json')), qualified_rate=review['qualified_rate'],
                      development_ready=review['development_ready'], created_utc=utc())
        write_json(RUNS/'reports'/(plan['name']+'-result.json'), record)
        if plan.get('development_only'):
            status('development_comparison_complete',candidate=plan['name'],checkpoint=str(checkpoint),
                   qualified_rate=review['qualified_rate'],development_ready=review['development_ready'])
            return
        if not review['development_ready']:
            status('needs_improvement', reason='Improvement candidate failed development criteria',
                   candidate=plan['name'], checkpoint=str(checkpoint), answers=str(raw),
                   qualified_rate=review['qualified_rate'])
            return
        for split in ('test', 'core', 'sampling'):
            prefix=plan.get('acceptance_prefix','50m-chat-v2')
            final_raw = RUNS/'reports'/f'{prefix}-{split}.json'
            if final_raw.exists():
                if read_json(final_raw)['checkpoint_sha256'] != sha256(checkpoint):
                    raise RuntimeError('Final suite already used for another candidate; create a new independent suite')
            else:
                run('eval.chat_suite', ['--checkpoint', checkpoint, '--split', split, '--output', final_raw,
                                       '--minutes', min(20, 1440-used())]+suite_options, prefix+'-final-'+split)
        status('awaiting_final_acceptance', checkpoint=str(checkpoint),
               base_checkpoint=plan['base_checkpoint'], candidate=plan['name'],
               acceptance_prefix=plan.get('acceptance_prefix','50m-chat-v2'),
               note='Review all test/core/sampling outputs before export')
    except ReviewPending:
        return
    except BaseException as exc:
        status('error', error=repr(exc), candidate=plan['name'])
        raise
    finally:
        owner.unlink(missing_ok=True)
        if awake.poll() is None:
            awake.terminate();awake.wait(timeout=10)


if __name__ == '__main__':
    main()
