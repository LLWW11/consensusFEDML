"""供PyCharm、VS Code等IDE直接点击运行的FedGCN入口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# PyCharm中直接运行本文件时，只需在这里切换默认配置。
# smoke_cpu：本机两轮CPU冒烟；server_cuda：服务器100轮CUDA正式训练。
DEFAULT_PROFILE = "server_cuda"

PROFILE_CONFIGS = {
    "smoke_cpu": "fedml_config_smoke_cpu.yaml",
    "server_cuda": "fedml_config_server_cuda.yaml",
}


def resolve_ide_profile(explicit_profile: Optional[str] = None) -> str:
    """解析显式参数、IDE环境变量或文件顶部默认配置指定的运行方案。"""

    profile = explicit_profile
    if profile is None:
        profile = os.environ.get("FEDGCN_IDE_PROFILE", DEFAULT_PROFILE)
    profile = str(profile).strip().lower()
    if profile not in PROFILE_CONFIGS:
        raise ValueError(
            "FEDGCN_IDE_PROFILE必须是 {}，实际为 {}".format(
                " 或 ".join(sorted(PROFILE_CONFIGS)), profile
            )
        )
    return profile


def prepare_fedml_arguments(profile: str) -> Path:
    """把IDE选择的配置转换成FedML命令行参数，并返回配置绝对路径。"""

    package_dir = Path(__file__).resolve().parent
    config_path = (package_dir / "configs" / PROFILE_CONFIGS[profile]).resolve()
    if not config_path.is_file():
        raise FileNotFoundError("找不到IDE运行配置：{}".format(config_path))
    sys.argv = [str(Path(__file__).resolve()), "--cf", str(config_path)]
    return config_path


def run_from_ide(profile: Optional[str] = None) -> None:
    """准备项目导入路径和FedML配置，然后调用与终端完全相同的训练入口。"""

    project_root = Path(__file__).resolve().parent.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    selected_profile = resolve_ide_profile(profile)
    config_path = prepare_fedml_arguments(selected_profile)
    print("IDE运行方案：{}".format(selected_profile))
    print("FedML配置文件：{}".format(config_path))

    # 延迟导入保证从任意IDE工作目录直接运行本文件时都能找到项目包。
    from FedGCN_fedml.run_fedgcn import main

    main()


if __name__ == "__main__":
    run_from_ide()

