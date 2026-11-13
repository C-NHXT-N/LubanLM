"""Offline CPU learning lab using the book's native model and generation core."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

from model import LubanConfig, LubanLM
from inference.generate import complete

# Original teaching text, bundled with the code; no external corpus is needed.
TRAIN = ["Linux是一个内核。", "模型根据前文预测下一个字符。", "训练后保存权重。", "恢复训练还需要优化器状态。"]
DEV = ["训练模型需要文字。", "保存后可以恢复模型。"]
PROMPT = "Linux是"
TOTAL = 16


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int, default=TOTAL)
    args = p.parse_args()
    if not 1 <= args.stop_after <= TOTAL:
        p.error("--stop-after must be between 1 and 16 (absolute update count)")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("output must be empty; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261023)
    vocab = {s: i for i, s in enumerate(["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + sorted(set("".join(TRAIN + DEV))))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
    tokenizer.save(str(args.output / "tokenizer.json"))
    identity = hashlib.sha256(json.dumps([TRAIN, DEV, vocab], ensure_ascii=False).encode()).hexdigest()
    config = LubanConfig(name="reader-tiny", status="offline-learning-only", vocab_size=len(vocab),
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, context_length=64, tie_word_embeddings=True,
        linear_bias=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = LubanLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, foreach=False)
    sampler = torch.Generator().manual_seed(1234)
    step, history = 0, []

    def batch(text):
        ids = [1] + tokenizer.encode(text).ids + [2]
        return torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])

    def observe():
        model.eval()
        with torch.no_grad():
            losses = [float(F.cross_entropy(model(x).flatten(0, 1), y.flatten())) for x, y in map(batch, DEV)]
            output, ids = complete(model, tokenizer, PROMPT, max_new_tokens=16, amp=False)
        model.train()
        return dict(step=step, dev_loss=sum(losses)/len(losses), prompt=PROMPT, output=output, generated_ids=ids)

    if args.resume:
        # Load only this lab's own trusted local checkpoint.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["config"] != model.config_dict() or saved["data_identity"] != identity or saved["total_steps"] != TOTAL:
            raise ValueError("checkpoint/config/data mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        step, history = saved["step"], saved["history"]
        if args.stop_after <= step:
            raise ValueError("stop-after must be greater than saved step")
    else:
        write(args.output / "initial.json", observe())

    while step < args.stop_after:
        index = int(torch.randint(len(TRAIN), (1,), generator=sampler))
        x, y = batch(TRAIN[index])
        # Same next-token objective as the full trainer; no padding in this one-row batch.
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        history.append(dict(step=step, train_loss=float(loss.detach()), input_tokens=x.numel(), supervised_tokens=y.numel()))
        print(json.dumps(history[-1]), flush=True)

    torch.save(dict(config=model.config_dict(), model=model.state_dict(), optimizer=optimizer.state_dict(),
        step=step, total_steps=TOTAL, data_identity=identity, history=history,
        sampler_rng=sampler.get_state(), torch_rng=torch.get_rng_state()), args.output / "last.pt")
    result = observe()
    result.update(parameters=sum(p.numel() for p in model.parameters()), updates_this_run=step-(saved["step"] if args.resume else 0),
        input_tokens=sum(row["input_tokens"] for row in history), supervised_tokens=sum(row["supervised_tokens"] for row in history),
        seconds=time.perf_counter()-started, device="cpu", dtype="float32", total_steps=TOTAL,
        torch_version=torch.__version__, history=history)
    write(args.output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
