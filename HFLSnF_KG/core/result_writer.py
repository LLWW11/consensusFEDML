"""分层联邦实验的配置、指标、拓扑和汇总输出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, TextIO


class ExperimentResultWriter:
    """把逐轮指标和调度记录写入一个独立结果目录。"""

    def __init__(
        self,
        result_dir: Path,
        schedule_filename: str = "topology_schedule.jsonl",
    ):
        """创建结果目录并准备指定名称的逐轮调度输出文件。"""

        self.result_dir = Path(result_dir).expanduser().resolve()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.result_dir / "metrics.csv"
        schedule_filename = str(schedule_filename).strip()
        if (
            not schedule_filename
            or Path(schedule_filename).name != schedule_filename
            or not schedule_filename.endswith(".jsonl")
        ):
            raise ValueError("调度文件名必须是当前目录下的.jsonl文件")
        self.topology_path = self.result_dir / schedule_filename
        self._metrics_file: Optional[TextIO] = None
        self._metrics_writer: Optional[csv.DictWriter] = None
        self._metric_fields = None
        self._topology_file = self.topology_path.open(
            "w", encoding="utf-8", newline=""
        )

    def write_json(self, filename: str, payload: Dict[str, object]) -> Path:
        """以UTF-8格式写入可读JSON文件并返回文件路径。"""

        path = self.result_dir / str(filename)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        return path

    def write_metrics(self, row: Dict[str, object]) -> None:
        """写入一行逐轮指标，并在首次调用时固定CSV字段顺序。"""

        if self._metrics_writer is None:
            self._metric_fields = tuple(row.keys())
            self._metrics_file = self.metrics_path.open(
                "w", encoding="utf-8-sig", newline=""
            )
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=list(self._metric_fields)
            )
            self._metrics_writer.writeheader()
        elif tuple(row.keys()) != self._metric_fields:
            raise ValueError(
                "逐轮指标字段发生变化，期望{}，实际{}".format(
                    self._metric_fields, tuple(row.keys())
                )
            )
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    def write_topology(self, record: Dict[str, object]) -> None:
        """写入一行JSONL拓扑与实际贡献记录。"""

        self._topology_file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )
        self._topology_file.flush()

    def close(self) -> None:
        """关闭已经打开的指标和拓扑文件。"""

        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._metrics_writer = None
        if self._topology_file is not None:
            self._topology_file.close()
            self._topology_file = None

    def __enter__(self) -> "ExperimentResultWriter":
        """进入上下文管理器并返回当前结果写入器。"""

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """离开上下文管理器时无条件关闭输出文件。"""

        self.close()
