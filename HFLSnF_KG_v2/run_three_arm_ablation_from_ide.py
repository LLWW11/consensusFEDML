"""供PyCharm、VSCode直接点击运行的V2三臂消融入口。"""

from __future__ import annotations

import sys
from pathlib import Path


# 本机先保持validate，只检查配置和文件，不会启动200轮训练。
# 放到CUDA服务器后改成run，即可按A、B、C顺序连续训练并自动汇总。
DEFAULT_ACTION = "validate"
DEFAULT_ARM = "all"


def run_from_ide() -> None:
    """使用文件顶部常量调用与终端完全相同的三臂实验入口。"""

    project_root = Path(__file__).resolve().parent.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    # 延迟导入，确保IDE把工作目录设在任意位置时仍能找到V2包。
    from HFLSnF_KG_v2.run_three_arm_ablation import main

    arguments = ["--action", DEFAULT_ACTION]
    if DEFAULT_ACTION == "run":
        arguments.extend(["--arm", DEFAULT_ARM])
    main(arguments)


if __name__ == "__main__":
    run_from_ide()
