"""用均匀分数的独立算例核对标准评分、滚动窗口和采样过滤。"""
from scripts.common import DATA, RUNS, write_json
from types import SimpleNamespace
import math
import torch
from torch import nn
from tokenizers import Tokenizer
from eval.lm_harness import LubanHarness
from inference.generate import sample

class UniformLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(context_length=4, bos_token_id=1, vocab_size=vocab)
    def forward(self, ids):
        return torch.zeros((*ids.shape, self.config.vocab_size), device=ids.device)

def main():
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(DATA/'tokenizers/vocab-32000/tokenizer.json'))
    vocab = tokenizer.get_vocab_size()
    adapter = LubanHarness(UniformLM(vocab), tokenizer)
    value, greedy = adapter.score([1,3], [4,5])
    assert abs(value+2*math.log(vocab)) < 1e-4
    assert not greedy
    text = 'Linux 是一个操作系统，负责管理内存和进程。'
    count = len(tokenizer.encode(text).ids)
    assert count > adapter.model.config.context_length
    rolling = adapter.loglikelihood_rolling([SimpleNamespace(args=(text,))])[0]
    error = abs(rolling + count*math.log(vocab))
    assert error < 1e-4
    pair = adapter.loglikelihood([SimpleNamespace(args=('答案：', 'A'))])[0]
    target_count = adapter.trace[-1]['answer_tokens']
    assert abs(pair[0] + target_count*math.log(vocab)) < 1e-4
    assert sample(torch.tensor([2.,0.,-1.]),[],temperature=0)==0
    assert sample(torch.tensor([2.,0.,-1.]),[],temperature=1,top_k=1)==0
    assert sample(torch.tensor([2.,0.,-1.]),[],temperature=1,top_p=.2)==0
    assert sample(torch.tensor([-1.,-1.5]),[0],repetition_penalty=2)==1
    write_json(RUNS/'reports/eval_checks.json',{'uniform_loglikelihood':value,'expected':-2*math.log(vocab),'rolling_tokens':count,'rolling_loglikelihood':rolling,'rolling_max_abs_error':error,'answer_boundary_tokens':target_count,'sampling_filters_and_negative_repetition_penalty':True,'passed':True})
    print('Independent uniform-score / rolling / sampling checks passed')

if __name__ == '__main__':
    main()
