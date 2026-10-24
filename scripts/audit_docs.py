"""检查新增正文编号与本地 Markdown 链接，不修改知识库。"""
from scripts.common import ROOT, RUNS, write_json
import re
from urllib.parse import unquote

def main():
    expected = [ROOT/f'C{chapter}/C{chapter}S{section}.md' for chapter in (4,5,6,7,8,12) for section in range(1,5)]
    expected += list((ROOT/'附录').glob('*.md'))
    numbered_count=len(expected)-len(list((ROOT/'附录').glob('*.md')))
    appendix_count=len(list((ROOT/'附录').glob('*.md')))
    expected += list((ROOT/'docs').glob('*.md'))
    expected += [ROOT/'章节正文要求.md',ROOT/'C3/C3S4.md']
    errors = []
    for path in expected:
        if not path.exists():
            errors.append('missing: '+str(path))
            continue
        text = path.read_text(encoding='utf-8-sig')
        if path.parent.name.startswith('C'):
            match = re.fullmatch(r'C(\d+)S(\d+)',path.stem)
            if not text.startswith(f'# {match[1]}.{match[2]} '):
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
