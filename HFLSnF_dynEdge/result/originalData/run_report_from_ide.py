"""供 VS Code、PyCharm 等 IDE 直接点击运行的实验报告入口。"""

from pathlib import Path
from typing import Tuple

from analyze_experiment_suite import run_analysis


# ==================== 只需修改这里 ====================
# 批次文件夹名对应本脚本所在目录的子目录
INPUT_BATCH_FOLDER = "varAlpha_0p1_client200_util0p6"
# 趋势和共识曲线使用的尾随平滑窗口。
SMOOTH_WINDOW = 10
# ======================================================


def resolve_ide_paths() -> Tuple[Path, Path]:
    """从本脚本固定位置推导项目根目录和输入批次目录。"""

    # 分析脚本集中在result/originalData，向上两级定位训练项目根目录。
    analysis_root = Path(__file__).resolve().parent
    project_root = analysis_root.parent.parent
    # 不读取 IDE 当前工作目录，批次直接取自分析脚本旁的同名子目录。
    input_dir = analysis_root / INPUT_BATCH_FOLDER
    return project_root, input_dir


def main() -> int:
    """运行完整分析；成功返回0，失败时打印简体中文错误并返回非零状态。"""

    project_root, input_dir = resolve_ide_paths()
    print("项目目录：{}".format(project_root))
    print("准备分析批次：{}".format(input_dir))
    try:
        output_dir = run_analysis(
            input_dir=input_dir,
            output_dir=None,
            smooth_window=SMOOTH_WINDOW,
        )
    except Exception as exc:  # IDE入口需要把数据、路径和绘图错误统一转成易读消息。
        print("报告生成失败：{}".format(exc))
        return 1

    print("IDE 一键运行完成。")
    print("结果目录：{}".format(output_dir.resolve()))
    print("分析报告：{}".format((output_dir / "分析报告.md").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
