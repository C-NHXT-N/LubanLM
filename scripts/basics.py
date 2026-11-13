"""附录 A 的可复算张量、梯度、Autograd 与交叉熵示例。"""
from scripts.common import RUNS, write_json
import torch
from torch.nn import functional as F

def main():
    x = torch.tensor([[1., 2.]])
    weight = torch.tensor([[2., 0.], [1., 3.]], requires_grad=True)
    product = x @ weight
    loss = (product.sum()-8)**2
    loss.backward()
    assert product.tolist() == [[4., 6.]]
    assert weight.grad.tolist() == [[4., 4.], [8., 8.]]
    logits = torch.tensor([[2., 0., -1.]])
    target = torch.tensor([0])
    cross_entropy = F.cross_entropy(logits, target)
    manual = -logits.softmax(-1)[0, 0].log()
    assert torch.allclose(cross_entropy, manual)
    result = {'input_shape': list(x.shape), 'weight_shape': list(weight.shape), 'matmul': product.tolist(), 'square_loss': float(loss), 'gradient': weight.grad.tolist(), 'probabilities': logits.softmax(-1).tolist(), 'cross_entropy': float(cross_entropy), 'manual_cross_entropy': float(manual)}
    write_json(RUNS / 'reports/basics.json', result)
    print(result)

if __name__ == '__main__':
    main()
