"""FedML设备选择和CUDA强制约束测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from FedGCN_fedml.device import resolve_fedml_device


class DeviceResolutionTest(unittest.TestCase):
    """验证本地CPU路径和服务器CUDA失败路径。"""

    def test_cpu_configuration_uses_fedml_selected_cpu(self):
        """本地配置应接受FedML返回的CPU设备。"""

        args = SimpleNamespace(using_gpu=False, require_cuda=False)
        device = resolve_fedml_device(
            args,
            fedml_device_getter=lambda _: torch.device("cpu"),
            cuda_available=False,
        )
        self.assertEqual(device, torch.device("cpu"))

    def test_required_cuda_fails_fast_without_gpu(self):
        """正式配置在无GPU环境中必须先于训练给出明确错误。"""

        args = SimpleNamespace(using_gpu=True, require_cuda=True)
        with self.assertRaisesRegex(RuntimeError, "正式配置要求CUDA"):
            resolve_fedml_device(
                args,
                fedml_device_getter=lambda _: torch.device("cpu"),
                cuda_available=False,
            )

    def test_require_cuda_also_requires_using_gpu(self):
        """互相矛盾的GPU开关不得进入FedML设备初始化。"""

        args = SimpleNamespace(using_gpu=False, require_cuda=True)
        with self.assertRaisesRegex(RuntimeError, "using_gpu=true"):
            resolve_fedml_device(
                args,
                fedml_device_getter=lambda _: torch.device("cpu"),
                cuda_available=False,
            )


if __name__ == "__main__":
    unittest.main()

