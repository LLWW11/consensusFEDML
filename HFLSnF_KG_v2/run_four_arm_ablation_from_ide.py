"""供PyCharm或VS Code直接点击运行的四臂二乘二入口。"""

from __future__ import annotations

import sys
from pathlib import Path


# 本机默认只校验，不会启动200轮正式训练。
# 放到CUDA服务器后改成"run-d"，即可只补跑D臂并复用三种目录中的A、B、C。
DEFAULT_ACTION = "validate"
EXISTING_THREE_ARM_ROOT = "results/三种"


def run_from_ide() -> None:
    """准备导入路径，并用文件顶部常量调用统一四臂入口。"""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from HFLSnF_KG_v2.run_four_arm_ablation import main

    arguments = ["--action", DEFAULT_ACTION]
    if DEFAULT_ACTION == "run-d":
        arguments.extend(
            [
                "--existing-root",
                str(
                    (
                        package_dir / EXISTING_THREE_ARM_ROOT
                    ).resolve()
                ),
            ]
        )
    main(arguments)


if __name__ == "__main__":
    run_from_ide()
