"""同一循环运行 50M/0.5B 预训练与 assistant-only SFT，保存可恢复检查点。"""
from __future__ import annotations
from scripts.common import DATA, RUNS, read_json, write_json, GPUBudget, require_gpu, sha256, utc
import os
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import LubanLM, LubanConfig
from training.batches import PackedTokens, FullPassTokens, SFTPairs, PrefetchedSFT
import shutil
from training.execution import loss_for, update, BatchTransfer, configure_model, configure_optimizer, validate_resume_settings, execution_settings
from training.v2_manifest import V2ManifestAdapter

def load_checkpoint(path, device='cpu'):
    # 只载入本项目在本地生成、用户指定的可信 checkpoint。
    return torch.load(path, map_location=device, weights_only=False)

@torch.no_grad()
def validation(model, dataset, size, device, dtype, batches=8, token_target=0, full_sft=False, details=False, include_counts=False):
    if size <= 0 or batches <= 0 or token_target < 0:
        raise ValueError('validation size/batches must be positive and token target nonnegative')
    was_training = model.training
    generator = torch.Generator().manual_seed(9090)
    state = dataset.state() if isinstance(dataset, SFTPairs) else None
    if state:
        old_generator = dataset.generator
        dataset.generator = generator
        dataset.order = list(range(len(dataset.rows)))
        dataset.cursor, dataset.epoch = 0, 0
    model.eval()
    numerator, tokens = 0.0, 0
    groups = {}
    if token_target and isinstance(dataset, PackedTokens):
        batches = math.ceil(token_target / (size * dataset.length))
    if full_sft and isinstance(dataset, SFTPairs):
        batches = math.ceil(len(dataset.rows) / size)
    try:
        for index in range(batches):
            count_rows = min(size, len(dataset.rows) - index * size) if full_sft and isinstance(dataset, SFTPairs) else size
            batch = dataset.batch(count_rows, generator)
            if token_target and isinstance(dataset, PackedTokens):
                remaining = token_target - tokens
                x, y, mask = batch[:3]
                valid_positions = torch.nonzero(y.reshape(-1) != -100, as_tuple=False).flatten()
                if valid_positions.numel() > remaining:
                    y = y.clone()
                    y.reshape(-1)[valid_positions[remaining:]] = -100
                    batch = (x, y, mask, *batch[3:])
            if details and isinstance(dataset,SFTPairs):
                x,y,mask = (v.to(device) for v in batch[:3])
                with torch.autocast('cuda',dtype=dtype):
                    logits = model(x,attention_mask=mask)
                    losses = F.cross_entropy(logits.float().flatten(0,1),y.flatten(),ignore_index=-100,reduction='none').reshape_as(y)
                sums = losses.sum(dim=1).cpu().tolist()
                counts = (y != -100).sum(dim=1).cpu().tolist()
                count = sum(counts)
                if count == 0:
                    continue
                loss = sum(sums)/count
                for row_index,(value,n) in enumerate(zip(sums,counts)):
                    if n == 0:
                        continue
                    source = dataset.rows[index*size+row_index].get('task','public')
                    aggregate = groups.setdefault(source,[0.,0])
                    aggregate[0] += value; aggregate[1] += n
            else:
                loss, _, count = loss_for(model, batch, device, dtype)
            numerator += float(loss) * count
            tokens += count
            if token_target and tokens >= token_target:
                break
    finally:
        if state:
            dataset.restore(state)
            dataset.generator = old_generator
        model.train(was_training)
    if tokens == 0:
        raise ValueError('validation produced no supervised tokens')
    value = numerator / tokens
    if details:
        sources = {k:dict(loss=v[0]/v[1],supervised_tokens=v[1]) for k,v in groups.items() if v[1]}
        result = dict(validation_loss=value,validation_tokens=tokens,by_task=sources,
                      macro_loss=sum(v['loss'] for v in sources.values())/len(sources))
        return result
    if include_counts:
        return dict(validation_loss=value,validation_tokens=tokens)
    return value

def schedule(step, total, base, warmup_steps=None):
    warmup = max(1, round(total * .05)) if warmup_steps is None else max(1, warmup_steps)
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return base * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=DATA, help='V1 dataset root; defaults to LUBAN_DATA')
    p.add_argument('--tokenizer', type=Path, help='explicit tokenizer.json path')
    p.add_argument('--v2-manifest', type=Path, help='adapt a V2 dataset_manifest.json for pretraining')
    p.add_argument('--stage', choices=['pretrain', 'sft'], default='pretrain')
    p.add_argument('--output', type=Path, default=RUNS / 'checkpoints/50m-base')
    p.add_argument('--init', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--continue-from', type=Path, help='new pretraining stage: carry weights and Adam state, reset sampler/schedule')
    p.add_argument('--minutes', type=float, default=130)
    p.add_argument('--tokens', type=int, default=8_000_000)
    p.add_argument('--steps', type=int)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--micro-batch', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--warmup-steps', type=int)
    p.add_argument('--validation-tokens', type=int, default=0)
    p.add_argument('--full-sft-validation', action='store_true')
    p.add_argument('--validation-every-tokens', type=int, default=0)
    p.add_argument('--checkpoint-tokens', type=int, default=0)
    p.add_argument('--save-every-minutes', type=float, default=0)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--pause-after-checkpoint', action='store_true')
    p.add_argument('--sft-train', type=Path)
    p.add_argument('--sft-validation', type=Path)
    p.add_argument('--execution', choices=['legacy', 'optimized'], default='legacy')
    p.add_argument('--attention-backend', choices=['manual', 'sdpa'], default='manual')
    p.add_argument('--optimizer-backend', choices=['loop', 'fused'], default='loop')
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--migrate-execution', action='store_true')
    p.add_argument('--validation-micro-batch', type=int)
    p.add_argument('--performance-reference-loss', type=float)
    p.add_argument('--sft-batching', choices=['legacy','flat','bucketed'],default='legacy')
    p.add_argument('--sft-sampling', choices=['random','stratified'],default='random')
    p.add_argument('--sft-objective',choices=['token','task-balanced'],default='token')
    p.add_argument('--prefetch-batches',type=int,choices=[0,2],default=0)
    p.add_argument('--source-validation',action='store_true')
    p.add_argument('--validation-every-steps',type=int,default=0)
    p.add_argument('--early-stop-patience',type=int,default=0)
    p.add_argument('--parent-sft-tokens',type=int)
    p.add_argument('--save-epoch-checkpoints',action='store_true',help='SFT: audit exact coverage and save each complete epoch')
    p.add_argument('--pretrain-sampling',choices=['random','full-pass'],default='full-pass',help='full-pass uses disjoint deterministic windows without replacement')
    p.add_argument('--sampler-seed',type=int,default=20261005)
    p.add_argument('--keep-step-checkpoints',type=int,default=0)
    p.add_argument('--checkpoint-bytes-cap',type=int,default=0)
    p.add_argument('--disk-free-bytes-floor',type=int,default=0)
    args = p.parse_args()
    args.data_root = args.data_root.resolve()
    args.tokenizer = (args.tokenizer or args.data_root / 'tokenizers/vocab-32000/tokenizer.json').resolve()
    if args.v2_manifest and args.stage != 'pretrain':
        p.error('--v2-manifest is currently supported for pretraining only')
    writer=write_json
    if args.save_epoch_checkpoints:
        from scripts.shortchat_io import write_json as writer
    args.validation_micro_batch = args.validation_micro_batch or args.micro_batch
    if args.save_epoch_checkpoints and args.stage != 'sft':
        p.error('--save-epoch-checkpoints requires SFT')
    if args.stage != 'sft' and any((args.sft_batching!='legacy',args.sft_sampling!='random',args.sft_objective!='token',args.source_validation)):
        p.error('SFT options require --stage sft')
    if args.sft_sampling == 'stratified' and (args.micro_batch!=8 or args.accumulation!=1 or args.sft_batching!='flat'):
        p.error('stratified v1 requires flat batching and 8x1')
    if args.sft_objective=='task-balanced' and args.sft_sampling!='stratified':
        p.error('task-balanced objective requires stratified sampling')
    if args.source_validation and not args.full_sft_validation:
        p.error('source validation requires full SFT validation')
    if min(args.validation_every_steps,args.early_stop_patience) < 0 or (args.parent_sft_tokens is not None and args.parent_sft_tokens<0):
        p.error('invalid SFT interval/lineage')
    if args.migrate_execution and not args.resume:
        p.error('--migrate-execution requires --resume')
    if args.validation_micro_batch <= 0 or (args.performance_reference_loss is not None and not math.isfinite(args.performance_reference_loss)):
        p.error('invalid validation execution settings')
    if sum(bool(x) for x in (args.init, args.resume, args.continue_from)) > 1:
        p.error('--init, --resume, --continue-from are mutually exclusive')
    if args.continue_from and args.stage != 'pretrain':
        p.error('--continue-from is only for pretraining; SFT uses --init')
    if args.log_every <= 0 or min(args.validation_tokens, args.validation_every_tokens, args.checkpoint_tokens, args.save_every_minutes) < 0 or (args.warmup_steps is not None and args.warmup_steps <= 0):
        p.error('invalid interval settings')
    if min(args.minutes, args.tokens, args.length, args.micro_batch, args.accumulation, args.lr, args.epochs) <= 0:
        p.error('training settings must be positive')
    if args.steps is not None and args.steps <= 0:
        p.error('--steps must be positive')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error('output contains files; choose a new output or resume')
    device = require_gpu()
    config = LubanConfig.from_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    generator = torch.Generator().manual_seed(2026)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with GPUBudget(f'{args.stage}-{config.name}', args.minutes) as budget:
        model = LubanLM(config).to(device)
        model.gradient_checkpointing = args.gradient_checkpointing
        configure_model(model, args.attention_backend)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1, betas=(.9, .95), foreach=False, fused=args.optimizer_backend == 'fused')
        scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
        if args.stage == 'pretrain':
            if args.v2_manifest:
                adapter = V2ManifestAdapter(args.v2_manifest, tokenizer_path=args.tokenizer)
                train_manifest = adapter.for_v1_reader('train')
                valid_manifest = adapter.for_v1_reader('dev')
                train = FullPassTokens(args.v2_manifest.parent,args.length,args.sampler_seed,args.tokens,manifest=train_manifest) if args.pretrain_sampling=='full-pass' else PackedTokens(args.v2_manifest.parent,args.length,manifest=train_manifest)
                valid = PackedTokens(args.v2_manifest.parent,args.length,manifest=valid_manifest)
            else:
                train = FullPassTokens(args.data_root/'processed/pretrain-train',args.length,args.sampler_seed,args.tokens) if args.pretrain_sampling=='full-pass' else PackedTokens(args.data_root/'processed/pretrain-train',args.length)
                valid = PackedTokens(args.data_root / 'processed/pretrain-validation', args.length)
            total = math.ceil(args.tokens / (args.micro_batch * args.accumulation * args.length))
            if isinstance(train,FullPassTokens): total=math.ceil(train.windows/(args.micro_batch*args.accumulation))
        else:
            train = SFTPairs(args.sft_train or args.data_root / 'processed/sft-train-tokens.jsonl', config.pad_token_id, generator,batching=args.sft_batching,sampling=args.sft_sampling)
            valid = SFTPairs(args.sft_validation or args.data_root / 'processed/sft-validation-tokens.jsonl', config.pad_token_id, torch.Generator().manual_seed(9090),batching='flat' if args.sft_batching!='legacy' else 'legacy')
            total = math.ceil(args.epochs * len(train.rows) / (args.micro_batch * args.accumulation))
        if args.steps:
            total = args.steps
        step, input_tokens, supervised_tokens, history = 0, 0, 0, []
        effective_input_tokens = 0
        counter_scope = 'entire_stage'
        parent = None
        parent_tokens = 0
        migrations = []
        parent_sft_tokens = 0
        best_score, bad_validations = float('inf'), 0
        train_identity = dict(train.identity)
        if args.stage == 'sft':
            train_identity['sha256'] = sha256(args.sft_train or args.data_root / 'processed/sft-train-tokens.jsonl')
            train_identity['validation_sha256'] = sha256(args.sft_validation or args.data_root / 'processed/sft-validation-tokens.jsonl')
        else:
            train_identity['validation_manifest'] = valid.identity
        tokenizer_hash = sha256(args.tokenizer)
        if args.init or args.resume or args.continue_from:
            state = load_checkpoint(args.resume or args.init or args.continue_from)
            if state['config'] != model.config_dict() or state['tokenizer_sha256'] != tokenizer_hash:
                raise ValueError('checkpoint architecture/tokenizer mismatch')
            model.load_state_dict(state['model'], strict=True)
            if args.init:
                inherited = state.get('cumulative_sft_tokens')
                if state['settings']['stage']=='pretrain':
                    inherited=0
                elif inherited is None:
                    inherited=args.parent_sft_tokens
                if inherited is not None and args.parent_sft_tokens is not None and args.parent_sft_tokens!=inherited:
                    raise ValueError('supplied SFT lineage conflicts with checkpoint')
                parent_sft_tokens=inherited
                parent={'path':str(args.init),'sha256':sha256(args.init),'step':state['step'],
                        'stage':state['settings']['stage'],'supervised_tokens':state['supervised_tokens'],
                        'inherited_sft_tokens':inherited,'lineage_complete':inherited is not None}
            if args.continue_from:
                if state['settings']['stage'] != 'pretrain':
                    raise ValueError('continue-from must reference a Base pretraining checkpoint')
                optimizer.load_state_dict(state['optimizer'])
                scaler.load_state_dict(state['scaler'])
                parent_tokens = state.get('cumulative_pretrain_tokens', state['supervised_tokens'])
                parent = {'path': str(args.continue_from), 'sha256': sha256(args.continue_from), 'step': state['step'], 'tokens': parent_tokens}
            if args.resume:
                settings = state['settings']
                validate_resume_settings(settings, vars(args), args.migrate_execution)
                if state['dataset'] != train_identity:
                    raise ValueError('resume dataset mismatch')
                for key,default in (('pretrain_sampling','random'),('sampler_seed',20261005)):
                    if settings.get(key,default)!=vars(args)[key]: raise ValueError(f'resume sampler setting changed: {key}')
                for key in ('warmup_steps', 'validation_tokens', 'full_sft_validation', 'validation_every_tokens', 'checkpoint_tokens', 'save_every_minutes'):
                    if settings.get(key, None if key == 'warmup_steps' else False if key == 'full_sft_validation' else 0) != vars(args)[key]:
                        raise ValueError(f'resume setting changed: {key}')
                for key in ('sft_train', 'sft_validation'):
                    if settings.get(key) != (str(vars(args)[key]) if vars(args)[key] else None):
                        raise ValueError(f'resume setting changed: {key}')
                for key,default in (('source_validation',False),('validation_every_steps',0),('early_stop_patience',0),('parent_sft_tokens',None),('save_epoch_checkpoints',False)):
                    if settings.get(key,default) != vars(args)[key]:
                        raise ValueError(f'resume setting changed: {key}')
                optimizer.load_state_dict(state['optimizer'])
                scaler.load_state_dict(state['scaler'])
                step, input_tokens, supervised_tokens = state['step'], state['input_tokens'], state['supervised_tokens']
                effective_input_tokens=state.get('effective_input_tokens',0)
                counter_scope=state.get('effective_input_token_scope','since_counter_upgrade' if step else 'entire_stage')
                history = state['history']
                parent, parent_tokens = state.get('parent'), state.get('parent_tokens', 0)
                parent_sft_tokens=state.get('parent_sft_tokens',0 if not parent else None)
                best_score=state.get('best_validation_score',float('inf'))
                bad_validations=state.get('bad_validations',0)
                migrations = state.get('execution_migrations', [])
                if args.migrate_execution:
                    migrations = migrations + [dict(checkpoint=str(args.resume),sha256=sha256(args.resume),step=step,created_utc=utc(),before={**execution_settings(settings),'micro_batch':settings['micro_batch'],'accumulation':settings['accumulation']},after={**execution_settings(vars(args)),'micro_batch':args.micro_batch,'accumulation':args.accumulation},bitwise_equivalence=False)]
                generator.set_state(state['data_rng'])
                torch.set_rng_state(state['torch_rng'])
                torch.cuda.set_rng_state_all(state['cuda_rng'])
                random.setstate(state['python_rng'])
                np.random.set_state(state['numpy_rng'])
                if isinstance(train, (SFTPairs,FullPassTokens)):
                    train.restore(state['sampler'])
        configure_optimizer(optimizer, args.optimizer_backend)
        order_audit = None
        if args.save_epoch_checkpoints:
            from training.sft_audit import SFTOrderAudit
            if len(train.rows) % (args.micro_batch * args.accumulation):
                raise ValueError('Epoch checkpoint audit requires whole effective batches')
            audit_state = state.get('sft_order_audit') if args.resume else None
            if args.resume and not audit_state:
                raise ValueError('Cannot resume an unaudited run with epoch audit enabled')
            order_audit = SFTOrderAudit(len(train.rows), audit_state)
        transfer = BatchTransfer(args.pin_memory)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        writer(args.output / 'run_config.json', {'settings': settings, 'config': model.config_dict(), 'dtype': str(dtype), 'gpu': torch.cuda.get_device_name(0), 'parameters': sum(p.numel() for p in model.parameters()), 'dataset': train_identity, 'tokenizer_sha256': tokenizer_hash, 'execution_migrations': migrations, 'restored_step': step, 'restored_input_tokens': input_tokens, 'restored_supervised_tokens': supervised_tokens,'training_seed':2026})
        def save(name):
            if args.disk_free_bytes_floor and shutil.disk_usage(args.output).free<args.disk_free_bytes_floor: raise RuntimeError('Disk reserve reached before checkpoint')
            state = {'model': model.state_dict(), 'config': model.config_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'step': step, 'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens, 'history': history, 'data_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'sampler': train.state() if isinstance(train, (SFTPairs,FullPassTokens)) else None, 'settings': settings, 'dataset': train_identity, 'tokenizer_sha256': tokenizer_hash, 'created_utc': utc(), 'parent': parent, 'parent_tokens': parent_tokens, 'cumulative_pretrain_tokens': parent_tokens + supervised_tokens if args.stage == 'pretrain' else None}
            state['execution_migrations'] = migrations
            if order_audit is not None:
                state['sft_order_audit'] = order_audit.state()
            state.update(effective_input_tokens=effective_input_tokens,effective_input_token_scope=counter_scope,
                         parent_sft_tokens=parent_sft_tokens,
                         cumulative_sft_tokens=parent_sft_tokens+supervised_tokens if args.stage=='sft' and parent_sft_tokens is not None else None,
                         best_validation_score=best_score,bad_validations=bad_validations)
            temporary = args.output / (name + '.tmp')
            torch.save(state, temporary)
            with temporary.open('r+b') as stream:os.fsync(stream.fileno())
            if args.save_epoch_checkpoints:
                from scripts.shortchat_io import replace
                replace(temporary,args.output/name)
            else:temporary.replace(args.output / name)
            if args.keep_step_checkpoints:
                old=sorted(args.output.glob('step-*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
                for path in old[args.keep_step_checkpoints:]: path.unlink()
            if args.checkpoint_bytes_cap and sum(p.stat().st_size for p in args.output.glob('*.pt'))>args.checkpoint_bytes_cap: raise RuntimeError('Checkpoint storage cap reached')
        model.train()
        def validate():
            # 原始 manual 路径 + 原验证微批次，使阶段验证数字保持可比。
            configure_model(model, 'manual')
            try:
                result=validation(model, valid, args.validation_micro_batch, device, dtype, token_target=args.validation_tokens, full_sft=args.full_sft_validation,details=args.source_validation,include_counts=True)
            finally:
                configure_model(model, args.attention_backend)
            value=result['validation_loss'] if isinstance(result,dict) else result
            record={'step':step,'validation_loss':value,'validation_tokens_target':args.validation_tokens,'full_sft':args.full_sft_validation,'created_utc':utc()}
            if isinstance(result,dict):record.update(result)
            writer(args.output / 'validation_latest.json',record)
            if args.source_validation:
                with (args.output/'validation_by_task.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record)+'\n')
            return value
        def consider_best():
            nonlocal best_score,bad_validations
            record=read_json(args.output/'validation_latest.json')
            score=record.get('macro_loss',record['validation_loss'])
            if score < best_score-.02:
                best_score,bad_validations=score,0
                save('best.pt')
                writer(args.output/'best_validation.json',dict(record,checkpoint='best.pt',selection='macro_task_loss' if args.source_validation else 'token_loss'))
            else:
                bad_validations+=1
        resumed_val = validate()
        initial_val = next((r['validation_loss'] for r in history if r.get('event')=='initial_validation'),resumed_val) if args.resume and args.save_epoch_checkpoints else resumed_val
        history.append({'step': step, 'event': 'resume_validation' if args.resume and args.save_epoch_checkpoints else 'initial_validation', 'validation_loss': resumed_val})
        if args.stage=='sft' and args.source_validation and not args.resume:
            consider_best()
        if not args.resume and not args.save_epoch_checkpoints:
            save('initial.pt')
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        start_tokens = input_tokens
        start_effective, start_supervised = effective_input_tokens, supervised_tokens
        interval = max(1, total // 4)
        validation_interval = max(1, math.ceil(args.validation_every_tokens / (args.micro_batch * args.accumulation * args.length))) if args.validation_every_tokens else interval
        if args.validation_every_steps:validation_interval=args.validation_every_steps
        checkpoint_interval = max(1, math.ceil(args.checkpoint_tokens / (args.micro_batch * args.accumulation * args.length))) if args.checkpoint_tokens else interval
        last_save = time.monotonic()
        stop_reason = 'step_target'
        prefetch=PrefetchedSFT(train,args.micro_batch,args.prefetch_batches) if args.prefetch_batches else None
        batch_source=prefetch or train
        try:
            while step < total:
                if args.disk_free_bytes_floor and shutil.disk_usage(args.output).free<args.disk_free_bytes_floor:
                    stop_reason='disk_reserve';break
                if args.performance_reference_loss is not None and history[-1].get('validation_loss', initial_val) > args.performance_reference_loss + .1:
                    stop_reason = 'performance_guard'
                    break
                if budget.remaining() < (180 if args.validation_tokens or args.full_sft_validation else 30):
                    stop_reason = 'time_budget'
                    break
                rate = schedule(step, total, args.lr, args.warmup_steps)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                batches = []
                consumed_indices = []
                for _ in range(args.accumulation):
                    count = args.micro_batch
                    if isinstance(train,FullPassTokens):
                        if not train.remaining: break
                        count=min(count,train.remaining)
                    if isinstance(train, SFTPairs):
                        epoch_size = len(train.order) if args.sft_sampling=='stratified' else len(train.rows)
                        remaining_examples = args.epochs * epoch_size - (train.epoch * epoch_size + train.cursor)
                        if remaining_examples <= 0:
                            break
                        count = min(count, remaining_examples)
                    batches.append(batch_source.batch(count, generator))
                    if order_audit is not None:
                        consumed_indices.extend(train.last_indices)
                if not batches:
                    stop_reason = 'epoch_target'
                    break
                effective_input_tokens += sum(int(b[2].sum()) if b[2] is not None else b[0].numel() for b in batches)
                batches = transfer.prepare(batches)
                loss_value, norm, inputs, targets = update(model, optimizer, scaler, batches, device, dtype, args.execution, args.pin_memory,objective=args.sft_objective)
                input_tokens += inputs
                supervised_tokens += targets
                step += 1
                epoch_complete = order_audit.add(consumed_indices, targets) if order_audit is not None else False
                if step == 1 or step % args.log_every == 0 or step % validation_interval == 0 or step == total:
                    torch.cuda.synchronize()
                    elapsed = max(time.monotonic() - start, 1e-9)
                    row = {'step': step, 'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens, 'loss': float(loss_value), 'lr': rate, 'grad_norm': float(norm), 'seconds': round(elapsed, 3), 'tokens_per_second': (input_tokens-start_tokens)/elapsed, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
                    row.update(effective_input_tokens=effective_input_tokens,effective_input_token_scope=counter_scope,
                               effective_tokens_per_second=(effective_input_tokens-start_effective)/elapsed,
                               supervised_tokens_per_second=(supervised_tokens-start_supervised)/elapsed,
                               input_token_scope='batch positions including padding')
                    if step % validation_interval == 0 or step == total:
                        row['validation_loss'] = validate()
                        if args.source_validation or args.stage=='pretrain':consider_best()
                    history.append(row)
                    with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                    if args.early_stop_patience and bad_validations>=args.early_stop_patience:
                        stop_reason='validation_patience'
                        break
                    if args.performance_reference_loss is not None and row.get('validation_loss', initial_val) > args.performance_reference_loss + .1:
                        stop_reason = 'performance_guard'
                        break
                if epoch_complete:
                    save(f'epoch-{len(order_audit.completed):02d}.pt')
                    writer(args.output / 'epoch_audit.json',dict(epochs=order_audit.completed,
                        supervised_tokens=order_audit.targets,examples_consumed=order_audit.consumed))
                    if args.pause_after_checkpoint and step < total:
                        stop_reason = 'checkpoint_review'
                        break
                if step % checkpoint_interval == 0 and not args.save_epoch_checkpoints:
                    save(f'step-{step:06d}.pt')
                    if args.pause_after_checkpoint and step < total:
                        stop_reason = 'checkpoint_review'
                        break
                if args.save_every_minutes and time.monotonic() - last_save >= args.save_every_minutes * 60:
                    save('latest.pt')
                    last_save = time.monotonic()
                if budget.tick():
                    stop_reason = 'time_budget'
                    break
        except BaseException as exc:
            stop_reason = 'error:' + type(exc).__name__
            raise
        finally:
            if prefetch:prefetch.close()
            final_val = validate()
            if args.source_validation and step % validation_interval:
                consider_best()
            history.append({'step': step, 'event': 'final_validation', 'validation_loss': final_val})
            save('last.pt')
            writer(args.output / 'summary.json', {'step': step, 'planned_steps': total, 'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens, 'initial_validation_loss': initial_val, 'final_validation_loss': final_val, 'stop_reason': stop_reason, 'elapsed_seconds': time.monotonic()-start, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'history': history, 'parent': parent, 'parent_tokens': parent_tokens, 'cumulative_pretrain_tokens': parent_tokens + supervised_tokens if args.stage == 'pretrain' else None, 'capability_status': 'not_accepted'})
            summary=read_json(args.output/'summary.json')
            summary.update(effective_input_tokens=effective_input_tokens,effective_input_token_scope=counter_scope,
                           parent_sft_tokens=parent_sft_tokens,
                           cumulative_sft_tokens=parent_sft_tokens+supervised_tokens if args.stage=='sft' and parent_sft_tokens is not None else None,
                           best_checkpoint=str(args.output/'best.pt') if (args.output/'best.pt').exists() else None)
            writer(args.output/'summary.json',summary)
            if args.stage == 'sft':
                train.rows.close()
                valid.rows.close()

if __name__ == '__main__':
    main()
