from scripts.common import read_json
from data.prepare import load_jsonl
from pathlib import Path
import numpy as np
import torch
import copy
import queue
import threading
import bisect

TASKS = ('public', 'greeting', 'identity', 'daily', 'concept', 'instruction', 'arithmetic')
TASK_WEIGHTS = (0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20)

class PackedTokens:
    def __init__(self, directory, length, *, manifest=None):
        manifest = manifest or read_json(Path(directory) / 'manifest.json')
        self.identity = manifest
        self.length = length
        self.shards = [np.memmap(row['path'], mode='r', dtype=manifest['dtype']) for row in manifest['shards']]
        self.starts = [max(0, len(s) - length) for s in self.shards]
        self.total_starts = sum(self.starts)
        self.cumulative_starts = cumulative64(self.starts)
        if self.total_starts == 0:
            raise ValueError('No complete token window')

    def batch(self, size, generator):
        rows = []
        for _ in range(size):
            position = int(torch.randint(self.total_starts, (1,), generator=generator))
            index = bisect.bisect_right(self.cumulative_starts,position)
            position -= self.cumulative_starts[index-1] if index else 0
            rows.append(torch.from_numpy(np.array(self.shards[index][position:position+self.length+1],dtype=np.int64)))
        ids = torch.stack(rows)
        return ids[:, :-1], ids[:, 1:], None

def cumulative64(values):
    values = [int(v) for v in values]
    if any(v < 0 for v in values): raise ValueError('Negative token extent')
    if sum(values)>np.iinfo(np.int64).max:raise ValueError('Token extent overflow')
    result = np.cumsum(values, dtype=np.int64).tolist()
    if result and (result[-1] != sum(values) or any(a > b for a,b in zip(result,result[1:]))):
        raise ValueError('Token extent overflow')
    return result

class FullPassTokens:
    """Disjoint windows; deterministic shuffled groups and windows, bounded state."""
    def __init__(self,directory,length,seed=20261005,token_limit=None,*,manifest=None):
        self.identity=manifest or read_json(Path(directory)/'manifest.json')
        self.length,self.seed,self.position=length,seed,0
        self.shards=[np.memmap(r['path'],mode='r',dtype=self.identity['dtype']) for r in self.identity['shards']]
        self.ends=cumulative64([len(s) for s in self.shards])
        if not self.ends or self.ends[-1] < 2: raise ValueError('Empty corpus')
        self.token_limit = min(self.ends[-1]-1, int(token_limit)) if token_limit else self.ends[-1]-1
        if self.token_limit <= 0: raise ValueError('Invalid token limit')
        self.identity = dict(self.identity, sampler_token_limit=self.token_limit)
        self.windows=(self.token_limit+length-1)//length
        self.group_size=4096
        self.groups=np.random.default_rng(seed).permutation((self.windows+self.group_size-1)//self.group_size)
        self.group_ends=cumulative64([min(self.group_size,self.windows-int(g)*self.group_size) for g in self.groups])
        self.generator=torch.Generator().manual_seed(seed)
        self.buffers=None
        self._group,self._order=None,None

    @property
    def remaining(self): return self.windows-self.position

    def state(self): return dict(position=self.position,seed=self.seed,length=self.length,windows=self.windows)

    def restore(self,state):
        if (state['seed'],state['length'],state['windows'])!=(self.seed,self.length,self.windows): raise ValueError('Full-pass sampler identity mismatch')
        if not 0<=state['position']<=self.windows: raise ValueError('Invalid full-pass cursor')
        self.position=state['position']

    def window_id(self,position):
        slot=bisect.bisect_right(self.group_ends,position)
        group=int(self.groups[slot]); within=position-(self.group_ends[slot-1] if slot else 0)
        if self._group!=group:
            count=min(self.group_size,self.windows-group*self.group_size)
            self._order=np.random.default_rng(np.random.SeedSequence([self.seed,group])).permutation(count)
            self._group=group
        return group*self.group_size+int(self._order[within])

    def read_window(self,start):
        result=np.zeros(self.length+1,dtype=np.int64)
        count=min(self.length+1,self.token_limit+1-start)
        offset=0
        while offset<count:
            index=bisect.bisect_right(self.ends,start+offset)
            local=start+offset-(self.ends[index-1] if index else 0)
            take=min(count-offset,len(self.shards[index])-local)
            result[offset:offset+take]=self.shards[index][local:local+take]
            offset+=take
        return result,count

    def batch(self,size,generator):
        if size>self.remaining: raise StopIteration('Full corpus consumed')
        rows,counts=[],[]
        for _ in range(size):
            row,count=self.read_window(self.window_id(self.position)*self.length)
            self.position+=1;rows.append(row);counts.append(count)
        ids=torch.from_numpy(np.stack(rows));labels=ids[:,1:].clone()
        mask=torch.zeros_like(labels,dtype=torch.bool)
        for i,count in enumerate(counts):
            labels[i,max(0,count-1):]=-100
            mask[i,:max(0,count-1)]=True
        return ids[:,:-1],labels,mask if any(n<self.length+1 for n in counts) else None

class SFTPairs:
    def __init__(self, path, pad_id, generator, *, batching='legacy', sampling='random'):
        if batching == 'legacy':
            self.rows = load_jsonl(path)
            versions = {r.get('encoding_version', 'legacy-joint-v1') for r in self.rows}
            self.lengths = np.array([len(r['input_ids']) for r in self.rows], dtype=np.int64)
            self.task_ids = [TASKS.index(r['task']) if r.get('task') in TASKS else 0 for r in self.rows]
        else:
            from training.sft_storage import MappedRows
            self.rows = MappedRows(path)
            versions = {self.rows.version}
            self.lengths = self.rows.lengths
            self.task_ids = [TASKS.index(task) if task in TASKS else 0 for task in self.rows.tasks]
            self.offsets = self.rows.offsets
            self.flat_ids,self.flat_labels = self.rows.ids,self.rows.labels
        self.identity = {'path': str(path), 'examples': len(self.rows)}
        if len(versions) != 1:
            raise ValueError('mixed SFT encoding versions')
        if versions != {'legacy-joint-v1'}:
            self.identity['encoding_version'] = next(iter(versions))
        self.batching, self.sampling = batching, sampling
        self.pad_id = pad_id
        self.generator = generator
        if sampling == 'stratified' and (len(set(self.task_ids)) != len(TASKS) or batching == 'bucketed'):
            raise ValueError('stratified sampling requires all seven tasks and flat/legacy batching')
        self.order = self.new_order()
        self.cursor = 0
        self.epoch = 0
        self.last_indices = []
        self.buffers = None

    def new_order(self):
        order = torch.randperm(len(self.rows), generator=self.generator).tolist()
        if self.sampling == 'stratified':
            groups = [[i for i in order if self.task_ids[i] == task] for task in range(len(TASKS))]
            # 每个完整批次2条公共数据+六类各1条；确定性循环少数类，消除物理重复。
            batches = max(1, (len(self.rows)+7)//8)
            order = []
            cursors = [0]*len(TASKS)
            for _ in range(batches):
                for task in (0,0,1,2,3,4,5,6):
                    group = groups[task]
                    order.append(group[cursors[task] % len(group)])
                    cursors[task] += 1
        elif self.batching == 'bucketed':
            batches = []
            for start in range(0, len(order), 128):
                window = sorted(order[start:start+128], key=lambda i:self.lengths[i])
                batches.extend(window[i:i+8] for i in range(0,len(window),8))
            permutation = torch.randperm(len(batches),generator=self.generator).tolist()
            order = [i for b in permutation for i in batches[b]]
        return order

    def state(self):
        return {'order': self.order, 'cursor': self.cursor, 'epoch': self.epoch}

    def restore(self, state):
        self.order, self.cursor, self.epoch = state['order'], state['cursor'], state['epoch']

    def batch(self, size, generator):
        selected = []
        for _ in range(size):
            if self.cursor >= len(self.order):
                self.order = self.new_order()
                self.cursor = 0
                self.epoch += 1
            selected.append(self.order[self.cursor])
            self.cursor += 1
        self.last_indices = selected[:]
        if self.batching != 'legacy':
            length = int(self.lengths[selected].max())
            if self.buffers is None or self.buffers[0].shape[0] < size or self.buffers[0].shape[1] < length:
                self.buffers = (torch.empty((size, max(length,512)),dtype=torch.long),
                                torch.empty((size, max(length,512)),dtype=torch.long),
                                torch.empty((size, max(length,512)),dtype=torch.bool))
            ids, labels, mask = (v[:size,:length] for v in self.buffers)
            ids.fill_(self.pad_id); labels.fill_(-100); mask.zero_()
            for i,index in enumerate(selected):
                start,end = self.offsets[index:index+2]
                count = int(end-start)
                ids[i,:count].copy_(torch.from_numpy(self.flat_ids[start:end].copy()))
                labels[i,:count].copy_(torch.from_numpy(self.flat_labels[start:end].copy()))
                mask[i,:count] = True
            # 独立结果保证累积槽/预取队列不会引用下次覆盖的组批缓冲区。
            batch = (ids[:,:-1].clone(),labels[:,1:].clone(),mask[:,:-1].clone())
            return batch + (torch.tensor([self.task_ids[i] for i in selected]),) if self.sampling == 'stratified' else batch
        selected = [self.rows[i] for i in selected]
        length = max(len(row['input_ids']) for row in selected)
        ids = torch.full((size, length), self.pad_id, dtype=torch.long)
        labels = torch.full((size, length), -100, dtype=torch.long)
        mask = torch.zeros((size, length), dtype=torch.bool)
        for i, row in enumerate(selected):
            count = len(row['input_ids'])
            ids[i, :count] = torch.tensor(row['input_ids'])
            labels[i, :count] = torch.tensor(row['labels'])
            mask[i, :count] = True
        return ids[:, :-1], labels[:, 1:], mask[:, :-1]


class PrefetchedSFT:
    """生产者使用私有采样器；检查点只保存消费者已经用掉的游标和RNG。"""
    def __init__(self, dataset, size, capacity=2):
        self.dataset, self.size = dataset, size
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.producer = copy.copy(dataset)
        self.producer.generator = torch.Generator()
        self.producer.generator.set_state(dataset.generator.get_state())
        self.producer.buffers = None
        self.thread = threading.Thread(target=self.produce,daemon=True)
        self.thread.start()

    def produce(self):
        try:
            while not self.stop.is_set():
                batch = self.producer.batch(self.size,self.producer.generator)
                entry = (batch,self.producer.state(),self.producer.generator.get_state(),self.producer.last_indices[:])
                while not self.stop.is_set():
                    try:
                        self.queue.put(entry,timeout=.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as exc:
            while not self.stop.is_set():
                try:
                    self.queue.put(exc,timeout=.1)
                    break
                except queue.Full:
                    pass

    def batch(self, size, generator):
        if size != self.size:
            # 末批先丢弃尚未消费的预取，再由已消费状态精确产生少量样本。
            self.close()
            return self.dataset.batch(size,generator)
        entry = self.queue.get()
        if isinstance(entry,BaseException):
            raise entry
        batch,state,rng,indices = entry
        self.dataset.restore(state)
        self.dataset.last_indices = indices
        generator.set_state(rng)
        return batch

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('SFT prefetch producer did not stop')
