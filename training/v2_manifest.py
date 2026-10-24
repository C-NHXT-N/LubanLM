"""Validate the simple V2 dataset manifest and adapt its train/dev shards to V1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.common import read_json, sha256


class V2ManifestAdapter:
    """Read V2 train/dev files without rewriting data or touching the frozen test split."""

    def __init__(self, manifest_path, *, tokenizer_path=None, eos_id=2, verify_hashes=True):
        self.path = Path(manifest_path).resolve()
        self.root = self.path.parent
        self.source = read_json(self.path)
        if self.source.get("pipeline_version") != "lubanlm-v2-minimal-data-v1":
            raise ValueError("unsupported V2 pipeline version")
        if self.source.get("tokenizer", {}).get("eos_id") != eos_id:
            raise ValueError("V2 manifest EOS ID mismatch")
        if tokenizer_path is not None:
            actual = sha256(tokenizer_path)
            if actual != self.source["tokenizer"].get("sha256"):
                raise ValueError("V2 manifest Tokenizer SHA-256 mismatch")
        collisions = self.source.get("split", {}).get("hash_collisions", {})
        expected_pairs = ("train:dev", "train:test", "dev:test")
        if any(collisions.get(pair) != 0 for pair in expected_pairs):
            raise ValueError("V2 manifest does not prove disjoint document splits")
        self.eos_id = int(eos_id)
        self.verify_hashes = verify_hashes
        self._splits = {}
        # Validate only training and development content. The frozen test file and
        # index are deliberately not opened by the training adapter.
        self._load_split("train")
        self._load_split("dev")
        if self._splits["train"]["doc_hashes"] & self._splits["dev"]["doc_hashes"]:
            raise ValueError("V2 train/dev document hashes overlap")

    def _inside_root(self, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"manifest path escapes dataset directory: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _load_split(self, split):
        if split in self._splits:
            return self._splits[split]
        if split not in ("train", "dev"):
            raise ValueError("training adapter supports only train and dev; test is frozen")
        record = self.source.get("outputs", {}).get(split)
        if not isinstance(record, dict):
            raise ValueError(f"V2 manifest has no {split} output")
        if record.get("dtype") != "uint16-le":
            raise ValueError(f"unsupported {split} dtype: {record.get('dtype')}")
        binary = self._inside_root(record["path"])
        index = self._inside_root(record["document_index_path"])
        token_count = int(record["tokens"])
        if token_count <= 1 or binary.stat().st_size != token_count * 2:
            raise ValueError(f"{split} binary token/byte count mismatch")
        if int(record.get("bytes", -1)) != binary.stat().st_size:
            raise ValueError(f"{split} manifest byte count mismatch")
        if self.verify_hashes:
            if sha256(binary) != record.get("sha256"):
                raise ValueError(f"{split} shard SHA-256 mismatch")
            if sha256(index) != record.get("document_index_sha256"):
                raise ValueError(f"{split} document-index SHA-256 mismatch")

        tokens = np.memmap(binary, mode="r", dtype="<u2")
        hashes = set()
        cursor = 0
        document_count = 0
        appended_eos = 0
        with index.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                digest = row.get("doc_hash")
                packed = int(row.get("packed_tokens", 0))
                start = int(row.get("start_token", -1))
                if row.get("split") != split or not digest or row.get("group_id") != digest:
                    raise ValueError(f"{split} index identity mismatch on line {line_number}")
                if digest in hashes:
                    raise ValueError(f"duplicate {split} document hash on line {line_number}")
                if start != cursor or packed < 1:
                    raise ValueError(f"{split} document offsets are not contiguous on line {line_number}")
                if row.get("appended_eos_id") != self.eos_id or row.get("body_eos_tokens") != 0:
                    raise ValueError(f"{split} document EOS metadata mismatch on line {line_number}")
                if int(tokens[start + packed - 1]) != self.eos_id:
                    raise ValueError(f"{split} EOS token missing at document boundary on line {line_number}")
                cursor += packed
                document_count += 1
                appended_eos += 1
                hashes.add(digest)
        del tokens
        if cursor != token_count or document_count != int(record.get("documents", -1)):
            raise ValueError(f"{split} index token/document totals mismatch")
        if appended_eos != int(record.get("appended_eos_count", -1)):
            raise ValueError(f"{split} appended EOS count mismatch")
        if int(record.get("target_tokens", -1)) != token_count - 1:
            raise ValueError(f"{split} next-token target count mismatch")

        identity = {
            "manifest_path": str(self.path),
            "manifest_sha256": sha256(self.path),
            "pipeline_version": self.source["pipeline_version"],
            "dataset_id": self.source.get("dataset_id"),
            "source": self.source.get("source", {}),
            "tokenizer_sha256": self.source["tokenizer"]["sha256"],
            "split": split,
            "tokens": token_count,
            "documents": document_count,
            "document_index_sha256": record["document_index_sha256"],
            "shard_sha256": record["sha256"],
            "eos_id": self.eos_id,
        }
        adapted = {
            "dtype": "<u2",
            "shards": [{"path": str(binary), "sha256": record["sha256"], "tokens": token_count}],
            "v2_identity": identity,
        }
        value = {"manifest": adapted, "doc_hashes": hashes}
        self._splits[split] = value
        return value

    def for_v1_reader(self, split):
        """Return the normalized manifest V1 PackedTokens/FullPassTokens already use."""
        return self._load_split(split)["manifest"]
