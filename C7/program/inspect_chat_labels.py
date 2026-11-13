"""Print every actual token and supervision label without loading model weights."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from scripts.linux_assistant.modeling import encode_training, render, messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    c = dict(system_prompt="简洁回答。", max_length=128)
    row = dict(id="teaching-pwd", messages=[dict(role="user", content="查看当前目录用什么命令？"), dict(role="assistant", content="pwd")])
    encoded = encode_training(c, tokenizer, row)
    output = dict(model=str(args.model.resolve()), sample=row, system_prompt=c["system_prompt"],
        text=render(tokenizer, messages(c, row), False), **encoded)
    output["tokens"] = [tokenizer.convert_ids_to_tokens(i) for i in encoded["input_ids"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(repr(output["text"]))
    for index, (token, token_id, label) in enumerate(zip(output["tokens"], output["input_ids"], output["labels"])):
        print(index, repr(token), token_id, label)


if __name__ == "__main__":
    main()
