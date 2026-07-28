"""运行阶段2集中式强TransE配方校准。"""

from __future__ import annotations

import json
import logging

import torch

from .core.device import resolve_fedml_device
from .experiment import (
    build_transe,
    checkpoint_payload,
    load_configured_dataset,
)
from .runtime import (
    create_result_directory,
    initialize_fedml_runtime,
    write_json,
)
from .tasks.kge import CentralizedTransETrainer


def main() -> None:
    """解析配置、训练集中式TransE并保存可审计结果。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    # 正式配置在读取22MB数据前验证CUDA，避免服务器误跑CPU。
    device = resolve_fedml_device(args)
    dataset = load_configured_dataset(args)
    model = build_transe(
        args,
        dataset.num_entities,
        dataset.num_relations,
    )
    result_dir = create_result_directory(args)
    write_json(
        result_dir / "config_snapshot.json",
        {
            key: value
            for key, value in sorted(vars(args).items())
        },
    )
    write_json(result_dir / "dataset_summary.json", dataset.summary())
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(result_dir / "relation2id.json", dataset.relation_to_id)

    trainer = CentralizedTransETrainer(
        args=args,
        dataset=dataset,
        model=model,
        device=device,
    )
    summary = trainer.train(result_dir)
    summary["device"] = str(device)
    summary["result_dir"] = str(result_dir)
    write_json(result_dir / "summary.json", summary)
    torch.save(
        checkpoint_payload(
            model.state_dict(),
            dataset,
            summary,
        ),
        result_dir / "model_best.pt",
    )
    logging.info(
        "V3集中式校准完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    print("V3集中式校准完成")
    print("结果目录：{}".format(result_dir))
    print(
        "完整测试MRR：{:.6f}".format(
            float(summary["final_test_metrics"]["mrr"])
        )
    )


if __name__ == "__main__":
    main()
