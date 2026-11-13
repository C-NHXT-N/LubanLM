"""python -m tokenizer.train 与 C1/program/1.3/train_tokenizer.py 参数相同。"""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    directory = Path(__file__).resolve().parents[1] / 'C1/program/1.3'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / 'train_tokenizer.py'), run_name='__main__')
