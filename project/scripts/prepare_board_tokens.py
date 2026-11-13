"""Reuse the exact PC test input token IDs, avoiding double template application."""
import json
from scripts.common import WORKSPACE,write_json

def main():
    original=json.loads((WORKSPACE/'reports/qwen-comparison-20261018-032232/question-outputs.json').read_text(encoding='utf8'))
    for name in ['Qwen3.5-2B','Qwen3-4B-Instruct-2507']:
        rows=[r for r in original['rows'] if r['model']==name]
        assert len(rows)==8
        folder=WORKSPACE/'exports/rk3576'/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'native-tokens.tsv').write_text(''.join(r['question_id']+'\t'+' '.join(map(str,r['input_ids']))+'\n' for r in rows),encoding='utf8')
        write_json(folder/'questions.json',[{'id':r['question_id'],'question':r['question'],'prompt_text':r['prompt_text'],'pc_output':r['output']} for r in rows])

if __name__=='__main__':main()
