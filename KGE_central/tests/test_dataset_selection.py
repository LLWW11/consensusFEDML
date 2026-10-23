"""验证双数据集配置、真实文件加载与集中式训练闭环。"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from KGE_central.configuration import PACKAGE_DIR, load_flat_config
from KGE_central.data import load_fb15k237, load_wn18rr
from KGE_central.evaluator import FilteredRankingEvaluator
from KGE_central.experiment import (
    _load_dataset,
    _state_dict_sha256,
    run_experiment,
)
from KGE_central.model import TransE


class DatasetSelectionTest(unittest.TestCase):
    """覆盖路径选择、文本边界以及最佳检查点的完整评估。"""

    def setUp(self) -> None:
        """为每个测试准备互不影响的临时数据和配置目录。"""

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _write_dataset(self) -> Path:
        """写入包含前导零和验证测试独有实体的小型词汇图谱。"""

        directory = self.root / "wn18rr"
        directory.mkdir(exist_ok=True)
        splits = {
            "train": (
                "0001\t_hypernym\t0002\n"
                "0002\t_hypernym\t0003\n"
                "0003\t_similar_to\t0001\n"
                "0001\t_similar_to\t0004\n"
            ),
            "valid": "0005\t_hypernym\t0001\n",
            "test": "0001\t_hypernym\t0006\n",
        }
        for split, content in splits.items():
            (directory / (split + ".txt")).write_text(content, encoding="utf-8")
        return directory

    def _write_config(self, dataset: str, data_dir=None) -> Path:
        """复用小模型冒烟配方，仅替换数据来源及结果名称。"""

        source = PACKAGE_DIR / "configs" / "smoke_synthetic_cpu.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["data_args"] = {"dataset": dataset}
        if data_dir is not None:
            config["data_args"]["data_dir"] = str(data_dir)
        # 小图也让每个epoch包含头、尾两种训练方向。
        config["train_args"]["batch_size"] = 2
        config["tracking_args"]["run_name"] = "central_{dataset}"
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_dataset_names_select_default_paths_and_run_names(self) -> None:
        """确认仅改名称即可选择正确默认目录和规范结果名称。"""

        for supplied, canonical, folder in (
            (" FB15K237 ", "fb15k-237", "FB15k-237"),
            ("fb15k-237", "fb15k-237", "FB15k-237"),
            ("WN18RR", "wn18rr", "wn18rr"),
        ):
            with self.subTest(dataset=supplied):
                config = load_flat_config(self._write_config(supplied))
                self.assertEqual(config["dataset"], canonical)
                self.assertEqual(config["run_name"], "central_" + canonical)
                expected = PACKAGE_DIR.parent / "HFLSnF_KG_v5" / "data" / folder
                self.assertEqual(Path(config["data_dir"]), expected.resolve())

    def test_explicit_paths_override_defaults_without_fallback(self) -> None:
        """确认绝对及相对路径优先，错误路径不会悄悄读取默认数据。"""

        actual = self._write_dataset()
        config = load_flat_config(self._write_config("wn18rr", actual))
        self.assertEqual(_load_dataset(config).num_entities, 6)
        config = load_flat_config(
            self._write_config("wn18rr", "tests/definitely_missing_dataset")
        )
        self.assertEqual(
            Path(config["data_dir"]),
            (PACKAGE_DIR / "tests/definitely_missing_dataset").resolve(),
        )
        with self.assertRaises(FileNotFoundError):
            _load_dataset(config)

    def test_unknown_dataset_is_rejected(self) -> None:
        """拒绝尚未支持的数据集，不能隐式走自由知识库加载分支。"""

        with self.assertRaisesRegex(ValueError, "dataset必须"):
            load_flat_config(self._write_config("db100k"))

    def test_synthetic_alias_still_works_without_data_directory(self) -> None:
        """保证既有合成数据别名和无外部文件的冒烟方式可用。"""

        config = load_flat_config(self._write_config("SYNTHETIC_KG"))
        self.assertEqual(config["dataset"], "synthetic-kg")
        self.assertEqual(_load_dataset(config).dataset_name, "synthetic-kg")

    def test_text_identifiers_and_evaluation_only_entities_are_preserved(self) -> None:
        """无字典文件时也保留前导零、首次出现编号及评估独有实体。"""

        directory = self._write_dataset()
        dataset = load_wn18rr(directory)
        self.assertEqual(dataset.dataset_name, "wn18rr")
        self.assertEqual(
            dataset.entity_to_id,
            {"0001": 0, "0002": 1, "0003": 2, "0004": 3, "0005": 4, "0006": 5},
        )
        self.assertEqual(dataset.num_relations, 2)
        self.assertEqual(dataset.valid_triples.tolist(), [[4, 0, 0]])
        self.assertEqual(dataset.test_triples.tolist(), [[0, 0, 5]])
        # 两个入口共享解析和编号，区别仅在数据集名称。
        reference = load_fb15k237(directory)
        self.assertEqual(reference.entity_to_id, dataset.entity_to_id)
        self.assertTrue(torch.equal(reference.train_triples, dataset.train_triples))

    def test_missing_empty_and_malformed_files_are_rejected(self) -> None:
        """对缺文件、空文件及非三列输入给出明确错误。"""

        directory = self._write_dataset()
        test_file = directory / "test.txt"
        test_file.unlink()
        with self.assertRaises(FileNotFoundError):
            load_wn18rr(directory)
        for content, message in (("", "不能为空"), ("0001 _hypernym\n", "三列")):
            test_file.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, message):
                    load_wn18rr(directory)

    def test_duplicates_within_and_between_splits_are_rejected(self) -> None:
        """保证扩展入口沿用划分内部和跨划分重复检查。"""

        directory = self._write_dataset()
        valid = directory / "valid.txt"
        for content in (
            "0005 _hypernym 0001\n0005 _hypernym 0001\n",
            "0001 _hypernym 0002\n",
        ):
            valid.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, "重复三元组"):
                    load_wn18rr(directory)

    def test_wn18rr_recipe_matches_existing_training_configuration(self) -> None:
        """确认首轮词汇数据实验只改变数据选择，保留已约定配方。"""

        configs = [
            load_flat_config(PACKAGE_DIR / "configs" / filename)
            for filename in (
                "centralized_fb15k237_strong_transe_cuda.yaml",
                "centralized_wn18rr_strong_transe_cuda.yaml",
            )
        ]
        excluded = {"dataset", "data_dir", "run_name", "config_file"}
        self.assertEqual(
            {key: value for key, value in configs[0].items() if key not in excluded},
            {key: value for key, value in configs[1].items() if key not in excluded},
        )
        self.assertEqual(configs[1]["dataset"], "wn18rr")

    @unittest.skipUnless(
        all(
            (PACKAGE_DIR.parent / "HFLSnF_KG_v5/data/wn18rr" / filename).is_file()
            for filename in ("train.txt", "valid.txt", "test.txt")
        ),
        "本机未安装WN18RR三份真实数据文件",
    )
    def test_real_wn18rr_statistics_and_full_validation_selection(self) -> None:
        """核对实际数据规模、训练未见实体和全量验证选择，不执行训练。"""

        config = load_flat_config(
            PACKAGE_DIR / "configs/centralized_wn18rr_strong_transe_cuda.yaml"
        )
        dataset = _load_dataset(config)
        self.assertEqual(dataset.num_entities, 40943)
        self.assertEqual(dataset.num_relations, 11)
        self.assertEqual(
            [int(triples.shape[0]) for triples in (
                dataset.train_triples, dataset.valid_triples, dataset.test_triples
            )],
            [86835, 3034, 3134],
        )
        self.assertEqual(len(dataset.all_true_triples), 93003)
        train_entities = set(dataset.train_triples[:, (0, 2)].reshape(-1).tolist())
        self.assertEqual(dataset.num_entities - len(train_entities), 384)
        selected = FilteredRankingEvaluator._select_triples(
            dataset.valid_triples, config["validation_max_triples"], 42, True
        )
        self.assertTrue(torch.equal(selected, dataset.valid_triples))

    @unittest.skipUnless(
        all(
            (PACKAGE_DIR.parent / version / "data/FB15k-237" / filename).is_file()
            for version in ("HFLSnF_KG_v3", "HFLSnF_KG_v5")
            for filename in ("train.txt", "valid.txt", "test.txt")
        ),
        "本机未同时安装旧目录和新默认目录的FB15k-237数据",
    )
    def test_real_fb15k237_default_matches_legacy_directory(self) -> None:
        """对照旧目录入口，验证默认路径迁移没有改变映射或三元组。"""

        config = load_flat_config(
            PACKAGE_DIR / "configs/centralized_fb15k237_strong_transe_cuda.yaml"
        )
        dataset = _load_dataset(config)
        old = load_fb15k237(PACKAGE_DIR.parent / "HFLSnF_KG_v3/data/FB15k-237")
        self.assertEqual(dataset.entity_to_id, old.entity_to_id)
        self.assertEqual(dataset.relation_to_id, old.relation_to_id)
        self.assertEqual(dataset.all_true_triples, old.all_true_triples)
        for split in ("train_triples", "valid_triples", "test_triples"):
            self.assertTrue(torch.equal(getattr(dataset, split), getattr(old, split)))

    def test_formal_configuration_refuses_cpu_before_writing_results(self) -> None:
        """正式配置无可用显卡时退出，不创建冒充正式训练的结果。"""

        output = self.root / "not_created"
        for filename in (
            "centralized_fb15k237_strong_transe_cuda.yaml",
            "centralized_wn18rr_strong_transe_cuda.yaml",
        ):
            with self.subTest(config=filename):
                with mock.patch("torch.cuda.is_available", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "拒绝降级"):
                        run_experiment(PACKAGE_DIR / "configs" / filename, output)
                self.assertFalse(output.exists())

    def test_wn18rr_cpu_training_saves_selected_model_and_complete_metrics(self) -> None:
        """用真实格式小图验证选模、检查点恢复、快照及完整头尾评估。"""

        path = self._write_config("WN18RR", self._write_dataset())
        result_dir, summary = run_experiment(path, self.root / "results")
        self.assertEqual(summary["dataset"], "wn18rr")
        self.assertTrue(result_dir.name.startswith("central_wn18rr_"))
        self.assertEqual(summary["epochs_ran"], 2)
        snapshot = json.loads((result_dir / "config_snapshot.json").read_text())
        self.assertEqual(snapshot["dataset"], "wn18rr")
        self.assertEqual(snapshot["run_name"], "central_wn18rr")
        self.assertEqual(Path(snapshot["data_dir"]), self.root / "wn18rr")
        with (result_dir / "metrics.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        selected = max(rows, key=lambda row: float(row["val_mrr"]))
        self.assertEqual(summary["best_epoch"], int(selected["epoch"]))
        self.assertEqual(
            summary["best_validation_mrr_during_training"], float(selected["val_mrr"])
        )
        for metric_group in ("final_validation_metrics", "final_test_metrics"):
            self.assertEqual(summary[metric_group]["evaluated_triple_count"], 1.0)
            self.assertEqual(summary[metric_group]["evaluated_query_count"], 2.0)
            self.assertTrue(all(math.isfinite(value) for value in summary[metric_group].values()))
        checkpoint = torch.load(str(result_dir / "model_best.pt"), map_location="cpu")
        dataset = load_wn18rr(self.root / "wn18rr")
        config = load_flat_config(path)
        model = TransE(dataset.num_entities, dataset.num_relations, config["embedding_dim"], config["distance_norm"])
        model.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(_state_dict_sha256(model), summary["best_model_hash"])
        restored = FilteredRankingEvaluator(dataset).evaluate(
            model, dataset.test_triples, torch.device("cpu")
        )
        self.assertEqual(restored, summary["final_test_metrics"])
        for filename in (
            "dataset_summary.json", "entity2id.json", "relation2id.json", "summary.json"
        ):
            self.assertTrue((result_dir / filename).is_file())

        # 用受控验证分数令最佳点固定在第一轮，独立复跑第一轮作参数对照。
        # 这样可以识别“报告最佳轮但实际保存末轮模型”的回归。
        with mock.patch(
            "KGE_central.trainer.CentralizedTransETrainer._evaluate_validation",
            side_effect=[
                {"mrr": 0.9, "hits_at_3": 1.0, "evaluated_triple_count": 1.0},
                {"mrr": 0.1, "hits_at_3": 1.0, "evaluated_triple_count": 1.0},
                {"mrr": 0.9},
            ],
        ):
            _, controlled = run_experiment(path, self.root / "controlled")
        self.assertEqual(controlled["best_epoch"], 1)
        one_epoch = yaml.safe_load(path.read_text(encoding="utf-8"))
        one_epoch["train_args"]["epochs"] = 1
        path.write_text(yaml.safe_dump(one_epoch), encoding="utf-8")
        _, reference = run_experiment(path, self.root / "one_epoch")
        self.assertEqual(controlled["best_model_hash"], reference["best_model_hash"])


if __name__ == "__main__":
    unittest.main()
