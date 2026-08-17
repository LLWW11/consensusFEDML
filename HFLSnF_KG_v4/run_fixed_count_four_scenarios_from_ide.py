"""供 IDE 直接修改动作并运行固定或动态四组对照实验。"""

from __future__ import annotations

from .run_fixed_count_four_scenarios import main


# Use "smoke" locally; use "dynamic150" or "formal150" on a CUDA server.
DEFAULT_ACTION = "smoke"
# Use "all" for every arm, or replace it with one matching arm name.
DEFAULT_ARM = "all"


def run_from_ide() -> None:
    """使用文件顶部常量调用固定或动态四组批量实验入口。"""

    main((DEFAULT_ACTION, "--arm", DEFAULT_ARM))


if __name__ == "__main__":
    run_from_ide()
