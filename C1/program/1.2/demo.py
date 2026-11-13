from pathlib import Path
from tempfile import TemporaryDirectory

from character_tokenizer import CharacterTokenizer
from teaching_bpe import TeachingBPE


def main() -> None:
    # 这两条短文本可让每一轮 BPE 合并都用手算复核。
    corpus = ["abab", "abac"]
    text = "abab"
    # 先观察字符级编码，再与 BPE 编码比较。
    character = CharacterTokenizer.train(corpus)
    print("字符词表大小:", len(character.vocabulary))
    print("字符 ID:", character.encode(text))
    print("字符还原:", character.decode(character.encode(text)))

    # 实际只有四轮可合并；12 只是上限，pair 耗尽后训练会停止。
    bpe = TeachingBPE.train(corpus, num_merges=12)
    print("BPE 每轮相邻对计数（显示前五项）:")
    for row in bpe.history:
        print(f"  第 {row['round']} 轮: {row['top_pairs']} → 合并 {row['chosen']}")
    print("BPE 合并规则:", bpe.merges)
    print("BPE 词表:", list(bpe.token_to_id))
    print("BPE Token:", bpe.encode_tokens(text))
    print("BPE Token ID:", bpe.encode(text))
    print("BPE 还原:", bpe.decode(bpe.encode(text)))
    # 临时保存并重载，确认合并规则不会因序列化丢失。
    with TemporaryDirectory(prefix="luban-bpe-demo-") as temporary:
        path = Path(temporary) / "teaching_bpe.json"
        bpe.save(path)
        restored = TeachingBPE.load(path)
        assert restored.encode(text) == bpe.encode(text)
    # 教学 BPE 不具备字节级回退能力，未见字符必须明确失败。
    try:
        bpe.encode("z")
    except ValueError as exc:
        print("未见字符检查:", exc)


if __name__ == "__main__":
    main()
