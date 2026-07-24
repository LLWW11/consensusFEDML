"""供PyCharm、VS Code等IDE直接点击运行的阶段二至阶段四入口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# PyCharm中直接运行本文件时，只需修改这里。
# smoke_cpu/server_cuda用于阶段二，transe_*用于阶段三，fedtranse_*用于阶段四。
DEFAULT_PROFILE = "transe_server_cuda"

PROFILE_CONFIGS = {
    "smoke_cpu": "smoke_cora_cpu.yaml",
    "server_cuda": "server_cora_cuda.yaml",
    "transe_smoke_cpu": "smoke_transe_synthetic_cpu.yaml",
    "transe_server_cuda": "server_fb15k237_transe_cuda.yaml",
    "fedtranse_smoke_cpu": "smoke_fedtranse_synthetic_cpu.yaml",
    "fedtranse_server_cuda": "server_fb15k237_fedtranse_cuda.yaml",
}


def resolve_ide_profile(explicit_profile: Optional[str] = None) -> str:
    """解析显式参数、IDE环境变量或文件顶部指定的运行方案。"""

    profile = explicit_profile
    if profile is None:
        profile = os.environ.get(
            "HFLSNF_KG_IDE_PROFILE", DEFAULT_PROFILE
        )
    profile = str(profile).strip().lower()
    if profile not in PROFILE_CONFIGS:
        raise ValueError(
            "HFLSNF_KG_IDE_PROFILE必须是{}，实际为{}".format(
                "或".join(sorted(PROFILE_CONFIGS)), profile
            )
        )
    return profile


def prepare_fedml_arguments(profile: str) -> Path:
    """把IDE运行方案转换成FedML命令行参数并返回配置路径。"""

    package_dir = Path(__file__).resolve().parent
    config_path = (
        package_dir / "configs" / PROFILE_CONFIGS[profile]
    ).resolve()
    if not config_path.is_file():
        raise FileNotFoundError("找不到IDE运行配置：{}".format(config_path))
    sys.argv = [str(Path(__file__).resolve()), "--cf", str(config_path)]
    return config_path


def resolve_entrypoint(profile: str):
    """根据IDE运行方案返回GCN、集中式或联邦TransE主函数。"""

    if str(profile).startswith("fedtranse_"):
        from HFLSnF_KG.run_federated_transe import main
    elif str(profile).startswith("transe_"):
        from HFLSnF_KG.run_transe import main
    else:
        from HFLSnF_KG.run_hfl_kg import main
    return main


def run_from_ide(profile: Optional[str] = None) -> None:
    """准备项目导入路径并调用与终端完全一致的训练入口。"""

    project_root = Path(__file__).resolve().parent.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    selected_profile = resolve_ide_profile(profile)
    config_path = prepare_fedml_arguments(selected_profile)
    print("IDE运行方案：{}".format(selected_profile))
    print("FedML配置文件：{}".format(config_path))

    # 延迟导入保证任意IDE工作目录都能找到项目包。
    main = resolve_entrypoint(selected_profile)
    main()


if __name__ == "__main__":
    run_from_ide()
