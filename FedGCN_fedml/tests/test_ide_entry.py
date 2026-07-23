"""IDE一键运行入口的轻量级测试。"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from FedGCN_fedml.run_from_ide import (
    prepare_fedml_arguments,
    resolve_ide_profile,
)


class IdeEntryTest(unittest.TestCase):
    """验证IDE配置选择和FedML参数构造不依赖工作目录。"""

    def test_default_profile_is_cpu_smoke(self):
        """没有环境变量时应默认选择本机CPU冒烟配置。"""

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_ide_profile(), "smoke_cpu")

    def test_environment_can_select_server_profile(self):
        """VS Code等IDE可以通过环境变量选择服务器CUDA配置。"""

        with mock.patch.dict(
            os.environ, {"FEDGCN_IDE_PROFILE": "server_cuda"}, clear=True
        ):
            self.assertEqual(resolve_ide_profile(), "server_cuda")

    def test_prepare_arguments_uses_absolute_config_path(self):
        """传给FedML的--cf路径必须是存在的绝对路径。"""

        original_argv = list(sys.argv)
        try:
            config_path = prepare_fedml_arguments("smoke_cpu")
            self.assertTrue(config_path.is_absolute())
            self.assertTrue(config_path.is_file())
            self.assertEqual(sys.argv[1], "--cf")
            self.assertEqual(sys.argv[2], str(config_path))
        finally:
            sys.argv = original_argv

    def test_unknown_profile_is_rejected(self):
        """未知IDE运行方案必须给出明确错误。"""

        with self.assertRaisesRegex(ValueError, "必须是"):
            resolve_ide_profile("unknown")


if __name__ == "__main__":
    unittest.main()

