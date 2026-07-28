"""供PyCharm或VS Code直接点击运行的评估桥接入口。"""

from __future__ import annotations

import sys
from pathlib import Path


# 默认只跑少量CPU查询，确认入口、检查点和输出文件没有问题。
# 放到CUDA服务器正式复评时改成"evaluation_bridge_full_cuda"。
DEFAULT_BRIDGE_PROFILE = "evaluation_bridge_smoke_cpu"


def main() -> None:
    """准备项目导入路径并调用统一IDE运行器。"""

    project_root = Path(__file__).resolve().parent.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from HFLSnF_KG_v2.run_from_ide import run_from_ide

    run_from_ide(DEFAULT_BRIDGE_PROFILE)


if __name__ == "__main__":
    main()
