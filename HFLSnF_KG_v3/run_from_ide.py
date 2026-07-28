"""为PyCharm和VS Code提供无需命令行参数的V3安全入口。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_PROFILE = "centralized_smoke_cpu"

PROFILE_TO_CONFIG = {
    "centralized_smoke_cpu": (
        "configs/smoke_centralized_cpu.yaml",
        "centralized",
    ),
    "centralized_strong_cuda": (
        "configs/centralized_fb15k237_strong_transe_cuda.yaml",
        "centralized",
    ),
    "hflsnf37_smoke_cpu": (
        "configs/smoke_hflsnf37_cpu.yaml",
        "hflsnf",
    ),
    "hflsnf37_strong_cuda": (
        "configs/hflsnf37_strong_transe_cuda.yaml",
        "hflsnf",
    ),
    "hflsnf37_benchmark_cuda": (
        "configs/benchmark_hflsnf37_accelerated_cuda.yaml",
        "hflsnf",
    ),
    "hflsnf37_profile_cuda": (
        "configs/benchmark_hflsnf37_profile_cuda.yaml",
        "hflsnf",
    ),
    "hflsnf37_row_count_benchmark_cuda": (
        "configs/benchmark_hflsnf37_row_count_cuda.yaml",
        "hflsnf",
    ),
    "hflsnf37_row_count_screen40_cuda": (
        "configs/screen_hflsnf37_row_count_seed42_40round_cuda.yaml",
        "hflsnf",
    ),
    "hflsnf37_varalpha0p1_formal300_cuda": (
        "configs/hflsnf37_row_count_varalpha0p1_seed42_300round_cuda.yaml",
        "hflsnf",
    ),
    "fixed37_fixed6_reset_adam_screen80_cuda": (
        "configs/screen_fixed37_fixed6_reset_adam_seed42_80round_cuda.yaml",
        "hflsnf",
    ),
    "fixed37_fixed6_persistent_adam_screen80_cuda": (
        "configs/screen_fixed37_fixed6_persistent_adam_seed42_80round_cuda.yaml",
        "hflsnf",
    ),
}


def resolve_profile(profile: str):
    """返回IDE方案对应的配置路径和入口类型。"""

    normalized = str(profile).strip()
    if normalized not in PROFILE_TO_CONFIG:
        raise ValueError(
            "未知V3 IDE方案{}；可选{}".format(
                normalized,
                "、".join(sorted(PROFILE_TO_CONFIG)),
            )
        )
    return PROFILE_TO_CONFIG[normalized]


def main() -> None:
    """设置FedML配置参数并调用集中式或HFLSnF入口。"""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    profile = os.environ.get(
        "HFLSNF_KG_V3_IDE_PROFILE",
        DEFAULT_PROFILE,
    )
    config_relative, entry_type = resolve_profile(profile)
    sys.argv = [
        str(Path(__file__).resolve()),
        "--cf",
        str((package_dir / config_relative).resolve()),
    ]
    if entry_type == "centralized":
        from HFLSnF_KG_v3.run_centralized_calibration import main as entry
    else:
        from HFLSnF_KG_v3.run_hflsnf37 import main as entry
    entry()


if __name__ == "__main__":
    main()
