"""IDE运行入口的配置选择和路径解析测试。"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from HFLSnF_KG import run_from_ide


class IdeEntryTest(unittest.TestCase):
    """验证IDE配置选择不会依赖当前工作目录。"""

    def test_default_profile_matches_file_setting(self) -> None:
        """验证未设置环境变量时采用文件顶部当前运行方案。"""

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                run_from_ide.resolve_ide_profile(),
                run_from_ide.DEFAULT_PROFILE,
            )

    def test_prepare_arguments_uses_existing_config(self) -> None:
        """验证IDE入口生成绝对YAML路径和FedML命令行参数。"""

        original_argv = list(sys.argv)
        try:
            config_path = run_from_ide.prepare_fedml_arguments("smoke_cpu")
            self.assertTrue(config_path.is_file())
            self.assertEqual(sys.argv[1], "--cf")
            self.assertEqual(sys.argv[2], str(config_path))
        finally:
            sys.argv = original_argv

    def test_transe_profile_selects_stage_three_entrypoint(self) -> None:
        """验证阶段三IDE方案选择TransE入口和现有配置文件。"""

        original_argv = list(sys.argv)
        try:
            config_path = run_from_ide.prepare_fedml_arguments(
                "transe_smoke_cpu"
            )
            entrypoint = run_from_ide.resolve_entrypoint(
                "transe_smoke_cpu"
            )
            self.assertEqual(entrypoint.__module__, "HFLSnF_KG.run_transe")
            self.assertTrue(config_path.is_file())
        finally:
            sys.argv = original_argv

    def test_fast_transe_profile_uses_existing_config(self) -> None:
        """验证4060加速方案可以从IDE解析到独立配置。"""

        original_argv = list(sys.argv)
        try:
            config_path = run_from_ide.prepare_fedml_arguments(
                "transe_server_cuda_fast"
            )
            self.assertTrue(config_path.is_file())
            self.assertEqual(
                config_path.name,
                "server_fb15k237_transe_cuda_fast.yaml",
            )
        finally:
            sys.argv = original_argv

    def test_fedtranse_profile_selects_stage_four_entrypoint(self) -> None:
        """验证阶段四IDE方案选择普通联邦TransE入口。"""

        original_argv = list(sys.argv)
        try:
            config_path = run_from_ide.prepare_fedml_arguments(
                "fedtranse_smoke_cpu"
            )
            entrypoint = run_from_ide.resolve_entrypoint(
                "fedtranse_smoke_cpu"
            )
            self.assertEqual(
                entrypoint.__module__,
                "HFLSnF_KG.run_federated_transe",
            )
            self.assertTrue(config_path.is_file())
        finally:
            sys.argv = original_argv

    def test_fixed_fedtranse_profiles_use_comparison_entrypoint(
        self,
    ) -> None:
        """验证四种固定拓扑IDE方案共用新的对照实验入口。"""

        original_argv = list(sys.argv)
        try:
            profiles = (
                "fixed_fedtranse_flnosnf_cuda",
                "fixed_fedtranse_flsnf_cuda",
                "fixed_fedtranse_hflnosnf_cuda",
                "fixed_fedtranse_hflsnf_cuda",
            )
            for profile in profiles:
                config_path = run_from_ide.prepare_fedml_arguments(
                    profile
                )
                entrypoint = run_from_ide.resolve_entrypoint(profile)
                self.assertTrue(config_path.is_file())
                self.assertEqual(
                    entrypoint.__module__,
                    "HFLSnF_KG.run_fixed_federated_transe",
                )
        finally:
            sys.argv = original_argv

    def test_dynamic_fedtranse_profile_uses_mat_entrypoint(
        self,
    ) -> None:
        """验证IDE动态方案选择MAT驱动联邦TransE入口。"""

        original_argv = list(sys.argv)
        try:
            profile = "dynamic_fedtranse_hflsnf_mat_cuda"
            config_path = run_from_ide.prepare_fedml_arguments(
                profile
            )
            entrypoint = run_from_ide.resolve_entrypoint(profile)
            self.assertTrue(config_path.is_file())
            self.assertEqual(
                config_path.name,
                "server_fb15k237_hflsnf_dynamic_mat_cuda.yaml",
            )
            self.assertEqual(
                entrypoint.__module__,
                "HFLSnF_KG.run_dynamic_federated_transe",
            )
        finally:
            sys.argv = original_argv

    def test_invalid_profile_is_rejected(self) -> None:
        """验证未知IDE运行方案给出明确错误。"""

        with self.assertRaisesRegex(ValueError, "必须是"):
            run_from_ide.resolve_ide_profile("unknown")


if __name__ == "__main__":
    unittest.main()
