"""C3 教学入口；核心实现集中在顶层 model 包。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from model.luban_lm import *

if __name__ == "__main__":
    main()
