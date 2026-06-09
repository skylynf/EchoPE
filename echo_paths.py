"""
EchoPrime 仓库根目录与运行环境初始化。

`utils` 等模块在导入时会读取相对路径下的 `assets/`、`model_data/` 等，
因此所有实验脚本应在 import utils / echo_prime 之前将工作目录切到仓库根
（与 README 中「在 EchoPrime 目录下运行」一致）。

本文件位于 EchoPrime/experiments/echo_paths.py，故 ECHO_ROOT = 上级目录。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_FILE = Path(__file__).resolve()
ECHO_ROOT = _FILE.parent.parent

# PE 等实验默认数据根：与 EchoPrime 仓库同级（例如 .../echoprime/EchoPrime 旁为 .../echoprime/hope_dataset）
DEFAULT_HOPE_DATASET = ECHO_ROOT.parent / "hope_dataset"


def setup_echo_root_cwd() -> Path:
    """切换到 ECHO_ROOT，并把仓库根与 utils 加入 sys.path（幂等）。"""
    os.chdir(ECHO_ROOT)
    for p in (str(ECHO_ROOT), str(ECHO_ROOT / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return ECHO_ROOT
