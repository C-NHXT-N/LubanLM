"""端到端小语料检查；不代表正式语料上的词表实验结果。"""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

# 从当前文件反推仓库根目录，避免依赖调用时所在的工作目录。
ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = ROOT / "C1" / "program" / "1.3"
# 将本章配套目录临时加入模块搜索路径，便于直接运行此检查脚本。
sys.path.insert(0, str(TRAIN_DIR))
from tokenizer_utils import encode_text, read_jsonl  # noqa: E402
sys.path.insert(0, str(ROOT / "C1" / "program" / "1.2"))
from character_tokenizer import CharacterTokenizer  # noqa: E402
from teaching_bpe import TeachingBPE  # noqa: E402


def run(*args: str) -> None:
    """用当前 Python 解释器运行子脚本；失败时立即把错误传给检查。"""
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    # 输入验证：空文件和非法 category 都应给出清楚错误。
    with tempfile.TemporaryDirectory(prefix="luban-invalid-data-") as temporary:
        empty_data = Path(temporary) / "empty.jsonl"
        empty_data.write_text("\n", encoding="utf-8")
        try:
            read_jsonl(empty_data)
            raise AssertionError("空数据集应报错")
        except ValueError as exc:
            assert "数据集为空" in str(exc)
        invalid_data = Path(temporary) / "invalid.jsonl"
        invalid_data.write_text('\n{"text":"x","category":"未知"}\n', encoding="utf-8")
        try:
            read_jsonl(invalid_data)
            raise AssertionError("非法类别应报错")
        except ValueError as exc:
            assert ":2:" in str(exc) and "category" in str(exc)

    # 基础算法：检查 BPE 首轮计数、确定性训练、保存重载和未见字符处理。
    toy_corpus = ["abab", "abac"]
    toy_bpe = TeachingBPE.train(toy_corpus, num_merges=4)
    assert toy_bpe.history[0]["top_pairs"][:3] == [("a", "b", 3), ("b", "a", 2), ("a", "c", 1)]
    assert toy_bpe.merges == TeachingBPE.train(toy_corpus, num_merges=4).merges
    assert toy_bpe.decode(toy_bpe.encode("abab")) == "abab"
    with tempfile.TemporaryDirectory(prefix="luban-teaching-") as temporary:
        path = Path(temporary) / "toy.json"
        toy_bpe.save(path)
        assert TeachingBPE.load(path).encode("abab") == toy_bpe.encode("abab")
    character = CharacterTokenizer.train(toy_corpus)
    assert character.decode(character.encode("abab")) == "abab"
    try:
        character.encode("z")
        raise AssertionError("字符 Tokenizer 应拒绝未见字符")
    except ValueError:
        pass

    # 端到端流程全部使用临时目录，运行后不留下模型或报告文件。
    with tempfile.TemporaryDirectory(prefix="luban-tokenizer-") as temporary:
        temp = Path(temporary)
        model_dir, report_dir = temp / "tokenizer", temp / "report"
        run(str(TRAIN_DIR / "train_tokenizer.py"), "--input", str(TRAIN_DIR / "sample_train.jsonl"), "--vocab-size", "320", "--output", str(model_dir))
        run(str(TRAIN_DIR / "evaluate_tokenizer.py"), "--tokenizer", str(model_dir), "--input", str(TRAIN_DIR / "sample_eval.jsonl"), "--report-dir", str(report_dir), "--hidden-size", "64")

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # 覆盖多语种、空白、空文本和较长输入，检查字节级往返还原。
        texts = ["中文", "English", "C++ 代码", "连续  空格\n缩进\t", "emoji 😀", "", "x" * 300]
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            assert tokenizer.decode(ids, skip_special_tokens=False) == text, repr(text)
        # Transformers 是可选依赖；安装后额外验证通用加载接口。
        if importlib.util.find_spec("transformers") is not None:
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            for text in texts:
                ids = hf_tokenizer.encode(text, add_special_tokens=False)
                assert hf_tokenizer.decode(ids, skip_special_tokens=False) == text, repr(text)
            print("AutoTokenizer 本地加载与往返检查通过。")
        else:
            print("未安装 Transformers：跳过可选 AutoTokenizer 检查。")
        # 边界符：空样本应得到 BOS/EOS，已有可见拼写时不应重复添加。
        ids = encode_text(tokenizer, "", add_bos=True, add_eos=True)
        assert ids == [tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")]
        bos_only = encode_text(tokenizer, "<bos>text", add_bos=True)
        assert bos_only[0] == tokenizer.token_to_id("<bos>") and bos_only.count(tokenizer.token_to_id("<bos>")) == 1
        eos_only = encode_text(tokenizer, "text<eos>", add_eos=True)
        assert eos_only[-1] == tokenizer.token_to_id("<eos>") and eos_only.count(tokenizer.token_to_id("<eos>")) == 1

        # 交付一致性：三个配置文件与评估报告必须指向同一份实际词表。
        config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        training = json.loads((model_dir / "training_config.json").read_text(encoding="utf-8"))
        report = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        assert training["actual_vocab_size"] == tokenizer.get_vocab_size(with_added_tokens=True)
        assert training["bytelevel_initial_alphabet_size"] == 256
        assert training["target_vocab_size"] == report["target_vocab_size"] == 320
        assert {"<pad>": 0, "<bos>": 1, "<eos>": 2} == training["special_token_ids"]
        assert config["vocab_size"] == report["vocab_size_including_special_tokens"]
        assert report["embedding_parameters"] == report["vocab_size_including_special_tokens"] * 64
        assert all(row["roundtrip_ok"] for row in report["by_category"].values())
    print("端到端检查通过：训练、导出、加载、边界符和评估报告均正常。")


if __name__ == "__main__":
    main()
