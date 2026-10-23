"""多数据集与物理拓扑选择的公共配置规则，不依赖FedML。"""

from pathlib import Path

from scipy.io import loadmat

from .tasks.kge.data import normalize_dataset_name


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGIES = {
    "metro": "../Topo_opt/postprocess/result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat",
    "open": "artifacts/topologies/open_varalpha0p1_reconstructed.mat",
}


def package_path(value):
    """将相对路径固定为本项目路径，不依赖终端当前目录。"""

    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else PACKAGE_DIR / path).resolve()


def apply_experiment_selection(args):
    """对显式多拓扑配置补全客户端数与数据目录，旧配置保持原行为。"""

    if not hasattr(args, "network_topology"):
        return args
    network = str(args.network_topology).strip().lower()
    selected = getattr(args, "dynamic_group_mat_file", None)
    if not selected:
        if network not in DEFAULT_TOPOLOGIES:
            raise ValueError("未知拓扑必须显式配置dynamic_group_mat_file")
        selected = DEFAULT_TOPOLOGIES[network]
    path = package_path(selected)
    if not path.is_file():
        raise FileNotFoundError("拓扑文件不存在，请先执行prepare：{}".format(path))
    metadata = loadmat(str(path), variable_names=["TopoOption", "num_of_nodes", "EdgeSet", "variance_control_analysis_only"], simplify_cells=True)
    if bool(metadata.get("variance_control_analysis_only", False)):
        raise ValueError("人数统计文件不能直接训练，请先执行prepare重建参与映射")
    actual = str(metadata.get("TopoOption", "")).strip().lower()
    if actual != network:
        raise ValueError("network_topology={}与MAT内拓扑{}不一致".format(network, actual))
    count = int(metadata["num_of_nodes"]) - 1
    # 客户端包含全部非云节点，随拓扑切换自动改变分区规模。
    for field in ("client_num_in_total", "client_num_per_round"):
        configured = getattr(args, field, None)
        if configured is not None and int(configured) != count:
            raise ValueError("{}必须匹配拓扑的{}个客户端".format(field, count))
        setattr(args, field, count)
    import numpy as np

    edge_count = int(np.asarray(metadata["EdgeSet"]).size)
    args.edge_num = 1 if str(args.topology_architecture).lower() == "fl" else edge_count
    args.network_topology = network
    args.dynamic_group_mat_file = str(path)
    args.dataset = normalize_dataset_name(args.dataset)
    if not getattr(args, "data_dir", None):
        folder = {"fb15k-237": "FB15k-237", "wn18rr": "wn18rr"}.get(args.dataset)
        if folder is not None:
            args.data_dir = str(PACKAGE_DIR / "data" / folder)
    replacements = {"dataset": args.dataset, "network_topology": network, "random_seed": str(args.random_seed)}
    for field in ("run_name", "result_root"):
        value = str(getattr(args, field))
        for name, replacement in replacements.items():
            value = value.replace("{" + name + "}", replacement)
        setattr(args, field, value)
    return args
