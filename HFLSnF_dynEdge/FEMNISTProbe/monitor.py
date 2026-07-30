"""在服务器训练期间定时记录 NVIDIA GPU 状态。"""

from __future__ import absolute_import

import csv
from datetime import datetime
import threading


class GPUMonitor:
    """使用 pynvml 在后台采集利用率、显存、温度、功率和时钟。"""

    COLUMNS = [
        "timestamp",
        "gpu_index",
        "gpu_name",
        "utilization_percent",
        "memory_used_mb",
        "memory_total_mb",
        "temperature_c",
        "power_w",
        "graphics_clock_mhz",
        "memory_clock_mhz",
    ]

    def __init__(self, output_path, gpu_index=0, interval_seconds=30):
        """初始化监控参数；缺少 NVML 时训练仍可继续。"""
        self.output_path = str(output_path)
        self.gpu_index = int(gpu_index)
        self.interval_seconds = max(1, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread = None
        self.available = False
        self.error_message = ""

    def _run(self):
        """在后台线程循环采集并追加 CSV。"""
        try:
            import pynvml  # pylint: disable=import-outside-toplevel
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            name_value = pynvml.nvmlDeviceGetName(handle)
            gpu_name = (
                name_value.decode("utf-8")
                if isinstance(name_value, bytes)
                else str(name_value)
            )
            self.available = True
            with open(
                    self.output_path, "w", encoding="utf-8", newline=""
            ) as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=self.COLUMNS)
                writer.writeheader()
                while not self._stop_event.is_set():
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    try:
                        power_w = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )
                    except pynvml.NVMLError:
                        power_w = ""
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "gpu_index": self.gpu_index,
                        "gpu_name": gpu_name,
                        "utilization_percent": int(utilization.gpu),
                        "memory_used_mb": memory.used / (1024.0 ** 2),
                        "memory_total_mb": memory.total / (1024.0 ** 2),
                        "temperature_c": pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        "power_w": power_w,
                        "graphics_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_GRAPHICS
                        ),
                        "memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_MEM
                        ),
                    }
                    writer.writerow(row)
                    file_obj.flush()
                    self._stop_event.wait(self.interval_seconds)
            pynvml.nvmlShutdown()
        except Exception as exc:  # pragma: no cover - 取决于服务器 NVML
            self.error_message = "{}: {}".format(type(exc).__name__, exc)

    def start(self):
        """启动后台GPU监控；重复调用不会创建第二个线程。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gpu-monitor", daemon=True
        )
        self._thread.start()

    def close(self):
        """停止并等待后台GPU监控线程。"""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 5)
        self._thread = None

    def __enter__(self):
        """进入上下文时启动监控。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文时停止监控。"""
        self.close()
        return False
