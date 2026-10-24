"""Check standalone tutorial contents and optionally verify its ZIP, using stdlib only."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

EXCLUDED_DIRS = {'.git', '.conda', '.venv', 'venv', 'env', '__pycache__', '.cache',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.revision-backups'}
ROOT_OUTPUTS = {'datasets', 'runs', 'provenance', 'archives', 'exports', 'reports', 'diagnostics'}
GENERATED = {'.pt', '.pth', '.ckpt', '.safetensors', '.rkllm', '.onnx', '.engine', '.bin',
    '.parquet', '.arrow', '.sqlite', '.db', '.pyc', '.pyo', '.zip', '.tmp', '.lock', '.sha256', '.log'}
TEXT = {'.md', '.py', '.pyw', '.json', '.jsonl', '.txt', '.ps1', '.sh', '.cpp', '.h', '.c', '.yaml', '.yml', '.toml', '.html', '.css', '.js'}
ALLOWED = TEXT | {'.svg', '.png', '.jpg', '.jpeg', '.webp', '.ipynb'}
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bAKIA[A-Z0-9]{16}\b|\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}')
LINK = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|[^)]+)\)')


def selected(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    if any(p in EXCLUDED_DIRS or p.startswith('.migration-staging') for p in parts):
        return False
    if parts[0] in ROOT_OUTPUTS or path.suffix.lower() in GENERATED:
        return False
    rel = '/'.join(parts)
    if re.search(r'(^|/)(data/(raw|processed|cache)|benchmark/results)(/|$)', rel):
        return False
    return True


def inspect(root: Path):
    files, errors = [], []
    paths = sorted((p for p in root.rglob('*') if p.is_file() and selected(p, root)), key=lambda p: p.as_posix())
    names = {p.relative_to(root).as_posix() for p in paths}
    markdown_count = link_count = python_count = 0
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append(dict(file=rel, reason='symlink_or_outside_root'))
            continue
        if path.suffix.lower() not in ALLOWED and path.name not in {'.gitignore', '.gitattributes', 'LICENSE', 'NOTICE'}:
            errors.append(dict(file=rel, reason='unsupported_release_file'))
        data = path.read_bytes()
        files.append(dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if path.suffix.lower() in TEXT or path.name in {'LICENSE', 'NOTICE'}:
            try:
                text = data.decode('utf-8-sig')
            except UnicodeDecodeError:
                errors.append(dict(file=rel, reason='not_utf8'))
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if SECRET.search(line):
                    errors.append(dict(file=rel, line=number, reason='credential_pattern'))
            if path.suffix in {'.py', '.pyw'}:
                python_count += 1
                try:
                    ast.parse(text, filename=rel)
                except SyntaxError as exc:
                    errors.append(dict(file=rel, line=exc.lineno, reason='python_syntax'))
            if path.suffix in {'.json', '.jsonl'}:
                try:
                    if path.suffix == '.json':
                        json.loads(text)
                    else:
                        for line in text.splitlines():
                            if line.strip():
                                json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(dict(file=rel, line=exc.lineno, reason='invalid_json'))
            if path.suffix == '.md':
                markdown_count += 1
                outside_code = re.sub(r'(`{3,})[^\n]*\n.*?\n\1', '', text, flags=re.S)
                for match in LINK.finditer(outside_code):
                    target = match.group(1).strip('<>')
                    if re.match(r'^[A-Za-z]:[/\\]', target) or target.startswith(('file:', '\\\\')):
                        errors.append(dict(file=rel, target=target, reason='machine_local_link'))
                        continue
                    if re.match(r'^[A-Za-z][\w+.-]*:', target) or target.startswith('//'):
                        continue
                    target = unquote(target.split('#')[0])
                    if not target:
                        continue
                    link_count += 1
                    resolved = (path.parent / target).resolve()
                    if not resolved.is_relative_to(root):
                        errors.append(dict(file=rel, target=target, reason='link_leaves_package'))
                    else:
                        name = resolved.relative_to(root).as_posix()
                        if name not in names and not any(n.startswith(name.rstrip('/')+'/') for n in names):
                            errors.append(dict(file=rel, target=target, reason='link_not_in_package'))
    return dict(tutorial='LubanLM', files=files, file_count=len(files), total_bytes=sum(f['bytes'] for f in files),
        markdown_files=markdown_count, local_links=link_count, python_files=python_count, errors=errors, passed=not errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--zip', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if args.output.resolve().is_relative_to(root):
        p.error('keep release audit outside tutorial sources')
    report = inspect(root)
    if args.zip:
        expected = {'LubanLM/'+f['path']: f for f in report['files']}
        with zipfile.ZipFile(args.zip) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected):
                report['errors'].append(dict(reason='zip_file_list_mismatch'))
            for name in set(names) & set(expected):
                if hashlib.sha256(archive.read(name)).hexdigest() != expected[name]['sha256']:
                    report['errors'].append(dict(file=name, reason='zip_sha256_mismatch'))
        report['zip_verified'] = not report['errors']
    report['passed'] = not report['errors']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
