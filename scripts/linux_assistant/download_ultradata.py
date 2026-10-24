"""Fetch a bounded, pinned UltraData source pool via a physical Windows adapter.

Credentials are read from the user's existing HF login before importing project
helpers. They are never written to reports or forwarded to storage redirects.
This entry does not publish records to the active dataset or approve training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import quote, urljoin, urlsplit

from huggingface_hub import get_token
from scripts.direct_mirror import PhysicalMirror

WORKSPACE = Path(__file__).resolve().parents[3]
AUTH_HOSTS = frozenset({"huggingface.co"})
ALLOWED_HOSTS = AUTH_HOSTS | frozenset({
    "hf-mirror.com", "dns.alidns.com", "cas-bridge.xethub.hf.co", "cdn-lfs.hf.co",
    "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    "cdn-lfs.hf-mirror.com", "us.aws.cdn.hf.co",
})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def workspace_path(value: str) -> Path:
    path = (WORKSPACE / value).resolve()
    path.relative_to(WORKSPACE)
    return path


class AuthenticatedPhysicalMirror(PhysicalMirror):
    def __init__(self, token: str, *args, authenticate_mirror=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._token = token
        self.auth_hosts = AUTH_HOSTS | ({"hf-mirror.com"} if authenticate_mirror else set())

    def open(self, url, headers=None):
        for _ in range(8):
            parts = urlsplit(url)
            if parts.scheme != "https" or parts.username or parts.port not in (None, 443):
                raise ValueError("Only HTTPS download URLs on approved hosts are allowed")
            self.check_host(parts.hostname)
            request_headers = {"User-Agent": "LubanLM-source-intake/1", "Accept-Encoding": "identity"}
            request_headers.update({k: v for k, v in (headers or {}).items() if k.lower() != "authorization"})
            # A storage redirect gets its own signed URL, never the HF account token.
            if parts.hostname in self.auth_hosts:
                request_headers["Authorization"] = "Bearer " + self._token
            connection = self.connection(parts.hostname, self.resolve(parts.hostname))
            try:
                path = parts.path + ("?" + parts.query if parts.query else "")
                connection.request("GET", path, headers=request_headers)
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    connection.close()
                    if not location:
                        raise ConnectionError("Redirect without Location")
                    url = urljoin(url, location)
                    continue
                return connection, response
            except BaseException:
                connection.close()
                raise
        raise ConnectionError("Too many download redirects")

    def fetch(self, url: str, path: Path, size: int, expected: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size == size and digest(path) == expected:
                return
            raise ValueError("Existing file does not match pinned upstream hash: " + path.name)
        part = path.with_suffix(path.suffix + ".part")
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError("Partial file exceeds pinned source size")
            if offset == size:
                if digest(part) != expected:
                    raise ValueError("Partial file has incorrect SHA256")
                part.replace(path)
                return
            connection = None
            try:
                connection, response = self.open(url, {"Range": f"bytes={offset}-"} if offset else {})
                self.record(event="file_response", file=path.name, status=response.status, offset=offset)
                if response.status in (401, 403):
                    raise PermissionError("Dataset access rejected: HTTP " + str(response.status))
                if response.status not in (200, 206):
                    raise ConnectionError("Download HTTP " + str(response.status))
                if response.status == 206 and not response.getheader("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid resume Content-Range")
                if offset and response.status == 200:
                    offset = 0
                with part.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(1 << 20):
                        offset += len(chunk)
                        if offset > size:
                            raise ValueError("Download exceeds pinned source size")
                        stream.write(chunk)
                if offset != size:
                    raise ConnectionError("Incomplete source file")
                if digest(part) != expected:
                    raise ValueError("Downloaded source SHA256 mismatch")
                part.replace(path)
                return
            except PermissionError:
                raise
            except (OSError, http.client.HTTPException) as exc:
                self.record(event="download_retry", file=path.name, attempt=attempt + 1, error=type(exc).__name__)
                if attempt == 5:
                    raise
                time.sleep(min(10, 2 ** attempt))
            finally:
                if connection:
                    connection.close()


def profile_file(path: Path, source_path: str) -> dict:
    keys, sources, source_fields, message_roles = Counter(), Counter(), Counter(), Counter()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty source row in {path.name}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row in {path.name}:{line_no}")
            keys.update(row.keys())
            count += 1
            for key in ("source", "dataset", "dataset_name", "source_dataset", "license", "id", "uid", "domain", "think_type"):
                if key in row:
                    source_fields[key] += 1
                    if key not in ("id", "uid") and isinstance(row[key], (str, int, float, bool)):
                        sources[key + ":" + str(row[key])[:200]] += 1
            for key in ("messages", "conversations"):
                messages = row.get(key)
                if isinstance(messages, list):
                    message_roles.update(str(m.get("role", m.get("from", "unknown"))) for m in messages if isinstance(m, dict))
    return {"source_path": source_path, "records": count, "fields": dict(keys),
            "provenance_field_counts": dict(source_fields), "source_labels": dict(sources.most_common(100)),
            "message_role_counts": dict(message_roles), "all_lines_parse_as_json_objects": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=WORKSPACE / "LubanLM/configs/linux-assistant-ultradata-download.json")
    parser.add_argument("--endpoint", choices=("https://hf-mirror.com", "https://huggingface.co"))
    parser.add_argument("--authenticate-mirror", action="store_true",
                        help="Send the saved token to hf-mirror.com only after explicit user authorization")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise RuntimeError("No saved personal HF login found; no anonymous access workaround will be attempted")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.endpoint:
        config["endpoint"] = args.endpoint
    report = workspace_path(config["report_dir"])
    provenance = workspace_path(config["provenance_dir"])
    raw = workspace_path(config["raw_dir"])
    report.mkdir(parents=True, exist_ok=True)
    transport = AuthenticatedPhysicalMirror(token, report / "network.jsonl", interface=config["interface"],
        allowed_hosts=ALLOWED_HOSTS, authenticate_mirror=args.authenticate_mirror)
    transport.endpoint = config["endpoint"]
    base = transport.endpoint + "/api/datasets/" + config["repo"]
    connection, response = transport.open(base + "/auth-check")
    try:
        auth_status = response.status
        response.read(65536)
    finally:
        connection.close()
    endpoint_authenticated = urlsplit(transport.endpoint).hostname in transport.auth_hosts
    write_json(report / "access-check.json", {"endpoint": transport.endpoint, "status": auth_status,
        "local_token_present": True, "credential_sent_to_endpoint": endpoint_authenticated,
        "credential_values_logged": False})
    if endpoint_authenticated and auth_status != 200:
        write_json(report / "download-progress.json", {
            "status": "blocked_repository_access", "endpoint": transport.endpoint,
            "access_check_status": auth_status, "repo": config["repo"], "revision": config["revision"],
            "planned_data_files": config["data_files"], "active_dataset_modified": False,
            "next_action": "Accept the repository access conditions using the same account as the saved HF login; then rerun this entry",
        })
        raise PermissionError("Dataset repository access check failed: HTTP " + str(auth_status))
    metadata = transport.metadata(config["repo"], config["revision"])
    write_json(provenance / "repository-metadata.json", metadata)
    blobs = {b["rfilename"]: b for b in metadata["siblings"]}
    total_bytes = sum(blobs[p]["size"] for p in config["data_files"] + config["documentation_files"])
    if total_bytes > config["max_download_bytes"]:
        raise ValueError("Selected source files exceed configured download budget")
    results = []
    for filename in config["documentation_files"] + config["data_files"]:
        blob = blobs[filename]
        path = workspace_path(config["raw_dir"] + "/" + filename) if filename in config["data_files"] else workspace_path(config["provenance_dir"] + "/" + filename)
        url = transport.endpoint + "/datasets/" + config["repo"] + "/resolve/" + config["revision"] + "/" + quote(filename, safe="/")
        if "lfs" in blob:
            transport.fetch(url, path, blob["size"], blob["lfs"]["sha256"])
            verification = "SHA256_matches_pinned_LFS_metadata"
        else:
            data = transport.bytes(url, maximum=1 << 20)
            git_hash = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            if git_hash != blob["blobId"] or len(data) != blob["size"]:
                raise ValueError("Documentation git blob identity mismatch: " + filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            verification = "git_blob_matches_pinned_metadata"
        result = {"upstream_path": filename, "local_path": path.relative_to(WORKSPACE).as_posix(), "bytes": path.stat().st_size,
                  "sha256": digest(path), "upstream_blob_id": blob["blobId"], "verification": verification}
        if filename in config["data_files"]:
            result["profile"] = profile_file(path, filename)
        results.append(result)
        write_json(report / "download-progress.json", {"status": "in_progress", "completed_files": len(results), "selected_files": len(config["data_files"]) + len(config["documentation_files"]), "files": results})
        print(json.dumps({"completed": filename, "bytes": result["bytes"], "records": result.get("profile", {}).get("records"), "hash_verified": True}), flush=True)
    manifest = {"status": "downloaded_raw_candidates_not_training_approved", "repo": config["repo"], "revision": config["revision"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "endpoint": transport.endpoint,
                "network_policy": "Physical WLAN adapter pinned; proxy/VPN/TUN not used; system network configuration unchanged",
                "physical_interface": transport.adapter, "access_check_status": auth_status,
                "credential_sent_to_mirror": args.authenticate_mirror and transport.endpoint == "https://hf-mirror.com",
                "mirror_credential_authorization": "Explicit user authorization in this conversation" if args.authenticate_mirror else None,
                "token_saved_or_logged": False, "selected_data_files": len(config["data_files"]),
                "total_bytes": sum(r["bytes"] for r in results), "total_records": sum(r.get("profile", {}).get("records", 0) for r in results),
                "selection_reason": config["selection_reason"], "files": results,
                "license_status": "Repository license and dataset cards preserved; selected records still need upstream-license and content review before publication",
                "training_approval": False, "active_dataset_modified": False}
    write_json(provenance / "download-manifest.json", manifest)
    write_json(report / "download-progress.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "selected_data_files", "total_bytes", "total_records", "active_dataset_modified")}), flush=True)


if __name__ == "__main__":
    main()
