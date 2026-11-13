"""可独立核验的训练执行路径；不改变数据、目标或模型参数格式。"""
import torch
from torch.nn import functional as F

EXECUTION_DEFAULTS = dict(execution='legacy', attention_backend='manual', optimizer_backend='loop', pin_memory=False)

def execution_settings(settings):
    return {key: settings.get(key, value) for key, value in EXECUTION_DEFAULTS.items()}

def configure_model(model, backend):
    for layer in model.model.layers:
        layer.self_attn.attention_backend = backend

def configure_optimizer(optimizer, backend):
    # load_state_dict 会覆盖构造参数；必须在加载后再设置执行选项。
    for group in optimizer.param_groups:
        group['foreach'] = False
        group['fused'] = backend == 'fused'
        if group['fused']:
            for parameter in group['params']:
                state = optimizer.state.get(parameter, {})
                if 'step' in state:
                    state['step'] = state['step'].to(parameter.device)
        else:
            for parameter in group['params']:
                state = optimizer.state.get(parameter, {})
                if 'step' in state:
                    state['step'] = state['step'].cpu()

def validate_resume_settings(old, new, migrate=False):
    for key in ('stage', 'length', 'lr', 'epochs', 'tokens', 'steps', 'gradient_checkpointing'):
        if old[key] != new[key]:
            raise ValueError(f'resume setting changed: {key}')
    for key,default in (('sft_batching','legacy'),('sft_sampling','random'),('sft_objective','token'),('prefetch_batches',0)):
        if old.get(key,default) != new.get(key,default):
            raise ValueError(f'resume SFT objective/sampler changed: {key}; start a new --init stage')
    batch_changed = any(old[key] != new[key] for key in ('micro_batch', 'accumulation'))
    backend_changed = execution_settings(old) != execution_settings(new)
    if (batch_changed or backend_changed) and not migrate:
        raise ValueError('resume execution changed; explicit --migrate-execution required')
    if old['micro_batch'] * old['accumulation'] != new['micro_batch'] * new['accumulation']:
        raise ValueError('execution migration must preserve effective batch')
    if old.get('validation_micro_batch', old['micro_batch']) != new['validation_micro_batch']:
        raise ValueError('resume validation micro batch changed')

class BatchTransfer:
    """每个累积槽使用独立 pinned 缓冲区；上次更新同步后才重用。"""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.buffers = {}

    def prepare(self, batches):
        if not self.enabled:
            return batches
        result = []
        for slot, batch in enumerate(batches):
            row = []
            for column, value in enumerate(batch):
                if value is None:
                    row.append(None)
                    continue
                # 每个槽只保留当前形状，SFT 变长输入不会无限占用 RAM。
                key = (slot, column)
                buffer = self.buffers.get(key)
                if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                    buffer = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                    self.buffers[key] = buffer
                buffer.copy_(value)
                row.append(buffer)
            result.append(tuple(row))
        return result

def trim_cuda_cache(device):
    """Release unused variable-shape blocks before WDDM starts paging them."""
    if not str(device).startswith('cuda'):
        return False
    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    unused = reserved - torch.cuda.memory_allocated(device)
    if reserved > total * .5 and unused > total * .15:
        torch.cuda.empty_cache()
        return True
    return False

def loss_for(model, batch, device, amp_dtype, *, legacy=False, non_blocking=False, row_weights=None):
    x, y, mask = batch[:3]
    count = int((y != -100).sum()) if not legacy else None
    x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
    mask = None if mask is None else mask.to(device, non_blocking=non_blocking)
    with torch.autocast('cuda', dtype=amp_dtype):
        logits = model(x, attention_mask=mask)
        if row_weights is None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        else:
            losses = F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape_as(y)
            counts = (y != -100).sum(dim=1).clamp_min(1)
            loss = ((losses.sum(dim=1)/counts)*row_weights.to(device,non_blocking=non_blocking)).sum()
    return loss, x.numel(), int((y != -100).sum()) if legacy else count

def update(model, optimizer, scaler, batches, device, dtype, execution='legacy', non_blocking=False, objective='token'):
    optimizer.zero_grad(set_to_none=True)
    trim_cuda_cache(device)
    target_count = sum(int((batch[1] != -100).sum()) for batch in batches)
    if target_count <= 0:
        raise ValueError('batch has no supervised tokens')
    weights = [None]*len(batches)
    if objective == 'task-balanced':
        from training.batches import TASK_WEIGHTS
        if any(len(batch) != 4 for batch in batches):
            raise ValueError('task-balanced loss requires explicit per-row task IDs')
        counts = torch.bincount(torch.cat([b[3] for b in batches]),minlength=len(TASK_WEIGHTS))
        active = sum(w for w,n in zip(TASK_WEIGHTS,counts) if int(n)>0)
        weights = [torch.tensor([TASK_WEIGHTS[int(task)]/int(counts[int(task)])/active for task in b[3]]) for b in batches]
    losses, inputs = 0., 0
    for batch, row_weights in zip(batches,weights):
        loss, count, targets = loss_for(model, batch, device, dtype, legacy=execution == 'legacy', non_blocking=non_blocking,row_weights=row_weights)
        if execution == 'legacy' and not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        scale = targets / target_count if row_weights is None else 1.
        scaler.scale(loss * scale).backward()
        losses += (float(loss.detach()) if execution == 'legacy' else loss.detach()) * scale
        inputs += count
    if execution != 'legacy' and not bool(torch.isfinite(losses)):
        raise RuntimeError('nonfinite loss; optimizer was not updated')
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    if execution == 'legacy':
        torch.cuda.synchronize()
        norm = float(norm)
    # 非有限检查/梯度裁剪在更新前同步，pinned CPU buffers 可安全重用。
    return losses, norm, inputs, target_count
