"""2.2 的单头注意力手算与未来位置扰动实验，仅依赖标准库。"""

import math


def softmax(values):
    """用减去最大值的写法计算 Softmax，避免 exp 溢出。"""
    # 统一平移不改变 Softmax 概率比例。
    shifted = [math.exp(value - max(values)) for value in values]
    return [value / sum(shifted) for value in shifted]


def attention(q, k, v, *, causal=True):
    """计算一个注意力头；q、k、v 的形状均为 [T, d]。"""
    # 每个 Token 的最后一维长度就是缩放分母中的 head_dim。
    d = len(q[0])
    outputs, all_weights = [], []
    for i, query in enumerate(q):
        # 当前 Query 与全部 Key 点积后按 sqrt(d) 缩放。
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(d) for key in k]
        if causal:
            # j > i 是未来位置；负无穷经过 Softmax 后权重恰为 0。
            scores = [score if j <= i else -math.inf for j, score in enumerate(scores)]
        weights = softmax(scores)
        # 同一组权重对所有 Value 维度加权求和，得到当前位置的输出向量。
        output = [sum(weights[j] * v[j][column] for j in range(len(v))) for column in range(d)]
        all_weights.append(weights)
        outputs.append(output)
    return all_weights, outputs


def rounded(matrix):
    """仅用于打印教学数值，不参与注意力计算。"""
    return [[round(value, 4) for value in row] for row in matrix]


def main():
    # 选单位 Q/K，使正文中的点积和 Softmax 可以手算。
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[1.0, 0.0], [0.0, 2.0]]
    weights, output = attention(q, k, v)
    print("有因果遮罩的权重:", rounded(weights))
    print("有因果遮罩的输出:", rounded(output))
    _, unmasked = attention(q, k, v, causal=False)
    print("无遮罩的输出:", rounded(unmasked))

    # 只修改位置 1（对位置 0 而言是未来）的 K/V，检验遮罩是否真的隔离未来。
    changed_k = [k[0], [100.0, 100.0]]
    changed_v = [v[0], [100.0, 100.0]]
    _, changed = attention(q, changed_k, changed_v)
    _, changed_unmasked = attention(q, changed_k, changed_v, causal=False)
    print("只改变未来 K/V 后，位置 0（有遮罩）:", rounded(changed)[0])
    print("只改变未来 K/V 后，位置 0（无遮罩）:", rounded(changed_unmasked)[0])
    # 有遮罩时位置 0 不受未来扰动；没有遮罩时输出应改变。
    assert output[0] == changed[0] == [1.0, 0.0]
    assert unmasked[0] != changed_unmasked[0]
    assert all(math.isclose(sum(row), 1.0) for row in weights)

    # 同一长度、精度和 head_dim 下，仅改变 KV 头数比较缓存体积。
    for name, kv_heads in [("MHA", 8), ("GQA", 2), ("MQA", 1)]:
        cache_bytes = 2 * 1 * 1 * 2048 * kv_heads * 64 * 2
        print(f"{name} 单层 KV Cache: {cache_bytes / 1024**2:g} MiB")


if __name__ == "__main__":
    main()
