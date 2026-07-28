"""供PyCharm或VS Code直接点击运行的无需重训方向诊断入口。"""

from __future__ import annotations

import sys
from pathlib import Path


# 默认只在CPU上评估8条测试事实，验证读取和输出流程。
# 服务器正式诊断时把FULL_CUDA改为True，并把D_RESULT_DIR填成D臂结果目录。
FULL_CUDA = False
EXISTING_RESULT_ROOT = "results/三种"
D_RESULT_DIR = ""


def run_from_ide() -> None:
    """根据文件顶部常量构造参数并调用统一终端入口。"""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from HFLSnF_KG_v2.run_directional_diagnostics import main

    arguments = [
        "--result-root",
        str((package_dir / EXISTING_RESULT_ROOT).resolve()),
    ]
    if D_RESULT_DIR.strip():
        d_path = Path(D_RESULT_DIR).expanduser()
        if not d_path.is_absolute():
            d_path = package_dir / d_path
        arguments.extend(
            [
                "--result",
                "dense_fede_fair={}".format(d_path.resolve()),
            ]
        )
    if FULL_CUDA:
        arguments.extend(
            [
                "--using-gpu",
                "--require-cuda",
                "--max-triples",
                "0",
            ]
        )
    else:
        arguments.extend(["--max-triples", "8"])
    main(arguments)


if __name__ == "__main__":
    run_from_ide()
