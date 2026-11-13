"""检查新增正文编号与本地 Markdown 链接，不修改知识库。"""
from scripts.common import REPO_ROOT
from scripts.common import ROOT, RUNS, write_json
import re
from urllib.parse import unquote

def main():
    expected = [REPO_ROOT/f'{chapter:02}/{section:02}.md' for chapter in range(13) for section in range(1,5)]
    expected += list((REPO_ROOT / '附录').glob('*.md'))
    numbered_count=len(expected)-len(list((REPO_ROOT / '附录').glob('*.md')))
    appendix_count=len(list((REPO_ROOT / '附录').glob('*.md')))
    expected += list((REPO_ROOT / 'docs').glob('*.md'))
    errors = []
    for path in expected:
        if not path.exists():
            errors.append('missing: '+str(path))
            continue
        text = path.read_text(encoding='utf-8-sig')
        if path.parent.name.isdigit():
            if not text.startswith(f'# {int(path.parent.name)}.{int(path.stem)} '):
                errors.append('heading: '+str(path))
        for target in re.findall(r'\]\(([^\n)]+)\)',text):
            if re.match(r'^[a-zA-Z]+://',target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0]).strip('<>')
            if target and not (path.parent/target).exists():
                errors.append(f'link: {path.name} -> {target}')
    result={'numbered_windows_sections':numbered_count,'appendix_files':appendix_count,'checked_documents':len(expected),'errors':errors,'passed':not errors}
    write_json(RUNS/'reports/document_checks.json',result)
    print(result)
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
