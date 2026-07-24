"""CPU设备和正式CUDA约束测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from HFLSnF_KG.core.device import resolve_fedml_device


class DeviceResolutionTest(unittest.TestCase):
    """验证本地CPU运行和服务器CUDA快速失败行为。"""

    def test_cpu_configuration_returns_cpu(self) -> None:
        """验证普通CPU配置可以正常解析。"""

        args = SimpleNamespace(using_gpu=False, require_cuda=False)
        device = resolve_fedml_device(
            args,
            fedml_device_getter=lambda _: "cpu",
            cuda_available=False,
        )
        self.assertEqual(device, torch.device("cpu"))

    def test_require_cuda_fails_without_gpu(self) -> None:
        """验证正式CUDA配置在无GPU机器上训练前报错。"""

        args = SimpleNamespace(using_gpu=True, require_cuda=True)
        with self.assertRaisesRegex(RuntimeError, "未检测到可用GPU"):
            resolve_fedml_device(
                args,
                fedml_device_getter=lambda _: "cpu",
                cuda_available=False,
            )


if __name__ == "__main__":
    unittest.main()
