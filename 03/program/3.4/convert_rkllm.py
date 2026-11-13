"""章节演示入口；复用 project 中的实现。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "project"))
from export.convert_rkllm import *

if __name__ == "__main__":
    main()
