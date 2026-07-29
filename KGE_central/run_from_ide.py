"""供IDE直接运行正式或CPU冒烟集中式配置。"""

from __future__ import annotations

from .configuration import PACKAGE_DIR
from .experiment import run_experiment


# Change to the strong CUDA filename on the GPU server.
DEFAULT_CONFIG_NAME = "smoke_synthetic_cpu.yaml"


def run_from_ide() -> None:
    """使用文件顶部配置名称运行集中式实验并打印结果目录。"""

    result_dir, summary = run_experiment(
        PACKAGE_DIR / "configs" / DEFAULT_CONFIG_NAME
    )
    print("结果目录：{}".format(result_dir))
    print(
        "测试MRR：{:.6f}".format(
            float(summary["final_test_metrics"]["mrr"])
        )
    )


if __name__ == "__main__":
    run_from_ide()
