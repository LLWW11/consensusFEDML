"""验证多数据集、可选语义域和OPEN受控调度接入。"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml
from scipy.io import loadmat

from HFLSnF_KG_v5.core.controlled_schedule import rebuild_groups
from HFLSnF_KG_v5.core.matlab_topology_schedule import MatlabTopologySchedule
from HFLSnF_KG_v5.experiment import build_federated_data, load_configured_dataset
from HFLSnF_KG_v5.multi_experiment import build_provider, load_configuration, model_hash, schedule_statistics
from HFLSnF_KG_v5.run_multi_topology_dataset import DEFAULT_CONFIGS, binding, check_finite, effective_configuration, main, run_batch, validate_result
from HFLSnF_KG_v5.selection import PACKAGE_DIR
from HFLSnF_KG_v5.tasks.kge.data import build_knowledge_graph_dataset, load_dataset
from HFLSnF_KG_v5.tasks.kge.graph_semantic_partition import relation_domains


def small_wordnet(heads=24):
    """构造足够多不可拆分包的词汇关系小图，无需真实大数据。"""

    train = [("{:08d}".format(index), relation, "{:08d}".format(100 + index % 5)) for index in range(heads) for relation in ("_hypernym", "_similar_to")]
    return build_knowledge_graph_dataset("wn18rr", train, [("00000901", "_hypernym", "00000100")], [("00000902", "_similar_to", "00000101")])


class MultiTopologyDatasetTest(unittest.TestCase):
    """检查旧接口兼容、真实拓扑尺度以及训练前的错误拒绝。"""

    def test_wn18rr_uses_relation_domains_and_preserves_packets(self):
        """关系域改变来源定义，域头包与负载算法保持原框架。"""

        dataset = small_wordnet()
        self.assertEqual(set(relation_domains(dataset).values()), {"_hypernym", "_similar_to"})
        args = SimpleNamespace(partition_strategy="semantic_domain_graph_local_balanced", partition_domain_extractor="auto", client_num_in_total=3, random_seed=42, partition_load_tolerance=0.1, partition_search_restarts=2)
        partition = build_federated_data(args, dataset)
        self.assertEqual(partition.summary()["domain_extractor"], "relation_name")
        owners = {}
        for client in partition.partitions:
            for head, relation, _ in client.train_triples.tolist():
                owners.setdefault((relation, head), set()).add(client.client_id)
        self.assertTrue(all(len(value) == 1 for value in owners.values()))
        self.assertEqual(sum(p.triple_count for p in partition.partitions), 48)
        args.partition_domain_extractor = "freebase_top_level"
        with self.assertRaisesRegex(ValueError, "WN18RR"):
            build_federated_data(args, dataset)

    def test_wn18rr_text_loading_keeps_leading_zeros_and_evaluation_entities(self):
        """真实文本读取不依赖字典，也不丢弃评估独有实体。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for split, text in (("train", "0001 _hypernym 0002\n"), ("valid", "0003 _hypernym 0001\n"), ("test", "0001 _hypernym 0004\n")):
                (path / (split + ".txt")).write_text(text, encoding="utf-8")
            dataset = load_configured_dataset(SimpleNamespace(dataset="WN18RR", data_dir=path))
            self.assertEqual(dataset.dataset_name, "wn18rr")
            self.assertEqual(dataset.entity_to_id, {"0001": 0, "0002": 1, "0003": 2, "0004": 3})
            with self.assertRaises(ValueError):
                load_dataset("db100k", path)

    def test_bad_explicit_data_path_and_duplicate_files_are_rejected(self):
        """显式错误路径不换数据源，缺文件与重复事实均报错。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_dataset("wn18rr", path / "missing")
            for split in ("train", "valid", "test"):
                (path / (split + ".txt")).write_text("0001 _hypernym 0002\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "训练集与验证集"):
                load_dataset("wn18rr", path)
            (path / "train.txt").write_text("0001 _hypernym 0002\n0001 _hypernym 0002\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复三元组"):
                load_dataset("wn18rr", path)

    def test_result_validation_checks_best_round_and_frozen_inputs(self):
        """使用模拟CUDA结果验证验收逻辑，不代表本机执行过GPU训练。"""

        entry = {"configuration": {"comm_round": 2, "dataset": "wn18rr"}, "partition_hash": "p", "initial_model_hash": "i", "schedule": {"schedule_hash": "s"}}
        summary = {"dataset": "wn18rr", "device": "cuda:0", "partition_hash": "p", "initial_model_hash": "i", "topology_schedule_hash": "s", "best_round": 2, "best_validation_mrr_during_training": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (path / "config_snapshot.json").write_text(json.dumps(entry["configuration"]), encoding="utf-8")
            (path / "model_best.pt").write_bytes(b"unit-test-checkpoint")
            keys = ["round", "mean_client_train_loss", "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10"]
            with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for index in (1, 2):
                    row = {key: 0.1 * index for key in keys}
                    row["round"] = index
                    writer.writerow(row)
            self.assertEqual(validate_result(path, entry)["best_round"], 2)
            summary["best_round"] = 1
            (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "最佳模型选择"):
                validate_result(path, entry)
            summary["best_round"] = 2
            summary["device"] = "cpu"
            (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是CUDA"):
                validate_result(path, entry)

    def test_reconstruction_preserves_counts_and_unique_legal_ids(self):
        """增减人数时保持源组关系，无重复并通过低参与者补位。"""

        clients = [1, 2, 3, 5]
        usage = {1: 10, 2: 2, 3: 0, 5: 0}
        priority = dict(zip(clients, range(4)))
        result = rebuild_groups([[1, 2], []], 3, clients, usage, priority)
        flat = [value for group in result for value in group]
        self.assertEqual(len(flat), 3)
        self.assertEqual(len(set(flat)), 3)
        self.assertIn(3, flat)
        result = rebuild_groups(result, 1, clients, usage, priority)
        self.assertEqual(sum(map(len, result)), 1)
        with self.assertRaises(ValueError):
            rebuild_groups([[1], [1]], 2, clients, usage, priority)
        with self.assertRaises(ValueError):
            rebuild_groups([[4], []], 1, clients, usage, priority)

    def test_default_three_configs_share_recipe_and_use_18_clients(self):
        """三方法按指定顺序运行，且沿用当前V5训练配方。"""

        configurations = [load_configuration(path) for path in DEFAULT_CONFIGS]
        self.assertEqual([args.ablation_arm for args in configurations], ["hflsnf", "hflnosnf", "flnosnf"])
        for args in configurations:
            self.assertEqual(args.client_num_in_total, 18)
            self.assertEqual(args.edge_num, 1 if args.ablation_arm == "flnosnf" else 4)
            self.assertEqual(args.dataset, "wn18rr")
            self.assertEqual(args.learning_rate, 5e-4)
            self.assertEqual(args.lr, 5e-4)
            self.assertEqual(args.comm_round, 150)
            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.embedding_dim, 256)
            self.assertEqual(args.server_optimizer, "fedavg")
            self.assertIn("open_wn18rr", args.run_name)
            self.assertNotIn("{", args.result_root)
            self.assertFalse(args.evaluate_test_after_training)

    def test_switching_network_and_dataset_derives_scale_and_paths(self):
        """仅改选择字段可切回37客户端城域网和自由知识库数据。"""

        source = yaml.safe_load(DEFAULT_CONFIGS[0].read_text(encoding="utf-8"))
        source["train_args"]["network_topology"] = "Metro"
        source["data_args"]["dataset"] = "FB15K237"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metro.yaml"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            args = load_configuration(path)
            self.assertEqual(args.client_num_in_total, 37)
            self.assertEqual(args.edge_num, 6)
            self.assertTrue(args.data_dir.endswith("FB15k-237"))
            self.assertEqual(args.dataset, "fb15k-237")
            source["train_args"]["client_num_in_total"] = 18
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "37"):
                load_configuration(path)

    def test_analysis_only_mat_cannot_be_loaded_as_training_schedule(self):
        """禁止将未重建映射的统计文件绕过检查直接用于训练。"""

        path = PACKAGE_DIR.parent / "Topo_other/OPEN/Result_JICN3/variance_control/jieguo_OPEN_epoch1000_varAlpha_0p1_analysis.mat"
        with self.assertRaisesRegex(ValueError, "映射未重建"):
            MatlabTopologySchedule(str(path), "hfl", True, "fixed", 0.6, client_num_in_total=18)

    def test_real_open_maps_exactly_match_controlled_counts(self):
        """三方法全1000轮人数一致，实际150轮覆盖全部客户端。"""

        source = loadmat(str(PACKAGE_DIR.parent / "Topo_other/OPEN/Result_JICN3/variance_control/jieguo_OPEN_epoch1000_varAlpha_0p1_analysis.mat"))
        fields = ("client_num_HFLSnF_fix", "client_num_HFLnoSnF", "client_num_FLnoSnF")
        partitions = [SimpleNamespace(client_id=index, triple_count=1) for index in range(18)]
        for config, field in zip(DEFAULT_CONFIGS, fields):
            args = load_configuration(config)
            provider = build_provider(args)
            expected = source[field + "_varctrl"][:, 5].tolist()
            self.assertEqual([provider.get_round(index).participant_count for index in range(1000)], expected)
            stats = schedule_statistics(provider, 150, partitions)
            self.assertTrue(all(value > 0 for value in stats["selection_counts"].values()))
            self.assertEqual(provider.describe()["physical_feasibility_validated"], 0)
            with self.assertRaises(IndexError):
                provider.get_round(1000)

    def test_model_hash_matches_actual_dynamic_trainer_format(self):
        """防止预检哈希与正式训练的模型指纹格式不一致。"""

        from HFLSnF_KG_v5.fedml_kge.dynamic_trainer import _state_dict_sha256
        model = torch.nn.Linear(2, 3)
        self.assertEqual(model_hash(model), _state_dict_sha256(model.state_dict()))

    def test_nonfinite_metrics_and_cuda_fallback_are_rejected(self):
        """出现非有限验证指标停止；无显卡时不复算或创建正式批次。"""

        row = {key: "0.1" for key in ("mean_client_train_loss", "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10")}
        row["round"] = "1"
        check_finite([row])
        row["val_mrr"] = "nan"
        with self.assertRaises(ValueError):
            check_finite([row])
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch("HFLSnF_KG_v5.run_multi_topology_dataset.validate_configurations") as validate:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                main(["formal3"])
            validate.assert_not_called()

    def test_three_methods_train_and_restore_on_tiny_wn18rr(self):
        """用真实OPEN映射完成CPU小图训练、验证选模及独立完整测试。"""

        from HFLSnF_KG_v5.core.randomness import seed_everything
        from HFLSnF_KG_v5.experiment import build_transe, checkpoint_payload
        from HFLSnF_KG_v5.fedml_kge import FedMLDynamicTopologyTransERunner
        from HFLSnF_KG_v5.tasks.kge.official_evaluation import run_best_checkpoint_official_evaluation

        dataset = small_wordnet(heads=36)
        partition = build_federated_data(load_configuration(DEFAULT_CONFIGS[0]), dataset)
        initial_hashes = set()
        with tempfile.TemporaryDirectory() as temporary:
            for config in DEFAULT_CONFIGS:
                args = load_configuration(config)
                # 冒烟预算仅覆盖临时小图，不写入或改变正式YAML。
                args.comm_round, args.epochs = 2, 1
                args.embedding_dim, args.negative_sample_count = 8, 4
                args.batch_size, args.evaluation_query_batch_size = 8, 2
                args.evaluation_candidate_batch_size = 16
                args.using_gpu, args.require_cuda = False, False
                seed_everything(args.random_seed)
                model = build_transe(args, dataset.num_entities, dataset.num_relations)
                initial_hashes.add(model_hash(model))
                provider = build_provider(args)
                runner = FedMLDynamicTopologyTransERunner(args, torch.device("cpu"), partition, model, provider)
                directory = Path(temporary) / args.ablation_arm
                summary = runner.run(directory)
                from HFLSnF_KG_v5.run_multi_topology_dataset import completed_metrics

                rows = completed_metrics(directory / "metrics.csv")
                check_finite(rows)
                self.assertEqual(len(rows), 2)
                self.assertEqual(summary["partition_hash"], partition.partition_hash)
                self.assertIn(summary["best_round"], (1, 2))
                self.assertIsNone(summary["final_test_metrics"])
                # 沿用正式入口的保存方式，检验恢复后的检查点可以完整排名。
                (directory / "config_snapshot.json").write_text(json.dumps(vars(args)), encoding="utf-8")
                torch.save(checkpoint_payload(runner.get_global_state(), dataset, summary, federated_data=partition), directory / "model_best.pt")
                contract = run_best_checkpoint_official_evaluation(dataset, directory, directory / "full_official_evaluation", torch.device("cpu"), query_batch_size=2, candidate_batch_size=16, progress_every=0)
                self.assertEqual(contract["combined_metrics"]["evaluated_query_count"], 2)
                self.assertTrue(math.isfinite(contract["combined_metrics"]["mrr"]))
        self.assertEqual(len(initial_hashes), 1)

    def test_batch_failure_resume_and_checkpoint_tampering(self):
        """模拟中途失败、跳过完成组及检查点改变时拒绝续用。"""

        with tempfile.TemporaryDirectory() as temporary:
            entries = []
            for arm in ("hflsnf", "hflnosnf", "flnosnf"):
                entries.append({"configuration": {"result_root": temporary, "ablation_arm": arm}, "config_sha256": arm, "data_sha256": {}, "mat_sha256": "mat", "partition_hash": "partition", "initial_model_hash": "model", "schedule": {"schedule_hash": arm}})
            validation = {"entries": entries}
            result = {"checkpoint_sha256": "original"}
            with mock.patch("HFLSnF_KG_v5.run_multi_topology_dataset.run_child", side_effect=[(Path(temporary) / "first", result), RuntimeError("模拟失败")]) as child:
                with self.assertRaisesRegex(RuntimeError, "模拟失败"):
                    run_batch(validation)
                self.assertEqual(child.call_count, 2)
            manifest_path = next(Path(temporary).glob("batch_*/batch_summary.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([entry["status"] for entry in manifest["entries"]], ["passed", "failed", "pending"])
            with mock.patch("HFLSnF_KG_v5.run_multi_topology_dataset.validate_result", return_value=result), mock.patch("HFLSnF_KG_v5.run_multi_topology_dataset.run_child", return_value=(Path(temporary) / "next", result)) as child:
                run_batch(validation, manifest_path)
                self.assertEqual(child.call_count, 2)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual([entry["attempts"] for entry in manifest["entries"]], [1, 2, 1])
            with mock.patch("HFLSnF_KG_v5.run_multi_topology_dataset.validate_result", return_value={"checkpoint_sha256": "changed"}):
                with self.assertRaisesRegex(ValueError, "检查点发生变化"):
                    run_batch(validation, manifest_path)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
