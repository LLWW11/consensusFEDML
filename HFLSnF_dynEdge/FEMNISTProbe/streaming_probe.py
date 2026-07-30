"""以后台线程把稀疏时间点的三层探针概率流式写入 HDF5。"""

from __future__ import absolute_import

from dataclasses import dataclass
import queue
import threading

import h5py
import numpy as np


@dataclass
class ProbeObservation:
    """保存一个探针时间点需要原子写入的全部数组。"""

    global_epoch: int
    topology_cycle_index: int
    mat_topology_index: int
    client_probabilities: np.ndarray
    edge_probabilities: np.ndarray
    cloud_probabilities: np.ndarray
    active_client_mask: np.ndarray
    edge_active_mask: np.ndarray


class StreamingProbeH5Writer:
    """使用单一后台线程拥有 HDF5 句柄，避免训练线程被压缩写入阻塞。"""

    _STOP = object()

    def __init__(
            self,
            output_path,
            observation_count,
            candidate_count,
            edge_slot_count,
            probe_labels,
            probe_indices,
            class_count,
            probe_hash,
            resume_count=0,
            compression_level=4,
    ):
        """创建固定形状的 HDF5 文件，或从已完成时间点继续写入。"""
        self.output_path = str(output_path)
        self.observation_count = int(observation_count)
        self.candidate_count = int(candidate_count)
        self.edge_slot_count = int(edge_slot_count)
        self.probe_labels = np.asarray(probe_labels, dtype=np.int64).reshape(-1)
        self.probe_indices = np.asarray(probe_indices, dtype=np.int64).reshape(-1)
        self.class_count = int(class_count)
        self.probe_hash = str(probe_hash)
        self.resume_count = int(resume_count)
        self.compression_level = int(compression_level)
        self._queue = queue.Queue(maxsize=2)
        self._error = None
        self._closed = False
        self._written_count = self.resume_count
        self._validate_arguments()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="femnist-probe-h5-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def written_count(self):
        """返回已经由后台线程完整写入的时间点数量。"""
        return int(self._written_count)

    def _validate_arguments(self):
        """校验HDF5维度和恢复位置。"""
        if self.observation_count <= 0:
            raise ValueError("observation_count 必须大于 0。")
        if self.candidate_count <= 0 or self.edge_slot_count <= 0:
            raise ValueError("候选数和边缘槽位数必须大于 0。")
        if self.probe_labels.size == 0:
            raise ValueError("探针标签不能为空。")
        if self.probe_labels.size != self.probe_indices.size:
            raise ValueError("探针标签和索引数量不一致。")
        if self.class_count < 2:
            raise ValueError("class_count 必须至少为 2。")
        if self.resume_count < 0 or self.resume_count > self.observation_count:
            raise ValueError("resume_count 超出可写时间点范围。")

    def _dataset_options(self, chunks):
        """返回统一的分块和压缩参数。"""
        return {
            "chunks": chunks,
            "compression": "gzip",
            "compression_opts": self.compression_level,
            "shuffle": True,
        }

    def _create_file(self):
        """创建新的固定形状HDF5及其坐标数据集。"""
        probe_count = int(self.probe_labels.size)
        with h5py.File(self.output_path, "w") as archive:
            archive.attrs["schema_version"] = "femnist_probe_h5_v1"
            archive.attrs["probe_hash"] = self.probe_hash
            archive.attrs["written_count"] = 0
            archive.create_dataset(
                "client_probabilities",
                shape=(
                    self.observation_count,
                    self.candidate_count,
                    probe_count,
                    self.class_count,
                ),
                dtype=np.float32,
                fillvalue=np.nan,
                **self._dataset_options((1, 1, probe_count, self.class_count))
            )
            archive.create_dataset(
                "edge_probabilities",
                shape=(
                    self.observation_count,
                    self.edge_slot_count,
                    probe_count,
                    self.class_count,
                ),
                dtype=np.float32,
                fillvalue=np.nan,
                **self._dataset_options((1, 1, probe_count, self.class_count))
            )
            archive.create_dataset(
                "cloud_probabilities",
                shape=(self.observation_count, probe_count, self.class_count),
                dtype=np.float32,
                fillvalue=np.nan,
                **self._dataset_options((1, probe_count, self.class_count))
            )
            archive.create_dataset(
                "active_client_mask",
                shape=(self.observation_count, self.candidate_count),
                dtype=np.bool_,
                fillvalue=False,
                chunks=(1, self.candidate_count),
            )
            archive.create_dataset(
                "edge_active_mask",
                shape=(self.observation_count, self.edge_slot_count),
                dtype=np.bool_,
                fillvalue=False,
                chunks=(1, self.edge_slot_count),
            )
            for name in [
                "global_epochs", "topology_cycle_indexes", "mat_topology_indexes"
            ]:
                archive.create_dataset(
                    name,
                    shape=(self.observation_count,),
                    dtype=np.int64,
                    fillvalue=-2,
                    chunks=(min(self.observation_count, 128),),
                )
            archive.create_dataset("probe_true_labels", data=self.probe_labels)
            archive.create_dataset("probe_indices", data=self.probe_indices)

    def _validate_existing_file(self):
        """校验恢复文件的固定坐标、形状和探针哈希。"""
        probe_count = int(self.probe_labels.size)
        expected_shapes = {
            "client_probabilities": (
                self.observation_count,
                self.candidate_count,
                probe_count,
                self.class_count,
            ),
            "edge_probabilities": (
                self.observation_count,
                self.edge_slot_count,
                probe_count,
                self.class_count,
            ),
            "cloud_probabilities": (
                self.observation_count, probe_count, self.class_count
            ),
        }
        with h5py.File(self.output_path, "r+") as archive:
            for name, expected_shape in expected_shapes.items():
                if name not in archive or archive[name].shape != expected_shape:
                    raise ValueError(
                        "恢复 HDF5 的 {} 形状错误：{}。".format(
                            name,
                            archive[name].shape if name in archive else None,
                        )
                    )
            if str(archive.attrs.get("probe_hash", "")) != self.probe_hash:
                raise ValueError("恢复 HDF5 的探针哈希不一致。")
            if not np.array_equal(
                    archive["probe_true_labels"][:], self.probe_labels
            ):
                raise ValueError("恢复 HDF5 的探针标签不一致。")
            if not np.array_equal(
                    archive["probe_indices"][:], self.probe_indices
            ):
                raise ValueError("恢复 HDF5 的探针索引不一致。")
            # 检查点是恢复真值；检查点之后可能存在未提交的探针行，允许覆盖。
            archive.attrs["written_count"] = self.resume_count
            archive.flush()

    def _write_observation(self, archive, observation):
        """把一个完整时间点写入预分配位置并更新提交计数。"""
        index = int(self._written_count)
        if index >= self.observation_count:
            raise IndexError("探针时间点数量超过预分配上限。")
        clients = np.asarray(observation.client_probabilities, dtype=np.float32)
        edges = np.asarray(observation.edge_probabilities, dtype=np.float32)
        cloud = np.asarray(observation.cloud_probabilities, dtype=np.float32)
        expected_clients = archive["client_probabilities"].shape[1:]
        expected_edges = archive["edge_probabilities"].shape[1:]
        expected_cloud = archive["cloud_probabilities"].shape[1:]
        if clients.shape != expected_clients:
            raise ValueError("客户端探针形状错误：{}。".format(clients.shape))
        if edges.shape != expected_edges:
            raise ValueError("边缘探针形状错误：{}。".format(edges.shape))
        if cloud.shape != expected_cloud:
            raise ValueError("云探针形状错误：{}。".format(cloud.shape))

        archive["client_probabilities"][index] = clients
        archive["edge_probabilities"][index] = edges
        archive["cloud_probabilities"][index] = cloud
        archive["active_client_mask"][index] = np.asarray(
            observation.active_client_mask, dtype=np.bool_
        )
        archive["edge_active_mask"][index] = np.asarray(
            observation.edge_active_mask, dtype=np.bool_
        )
        archive["global_epochs"][index] = int(observation.global_epoch)
        archive["topology_cycle_indexes"][index] = int(
            observation.topology_cycle_index
        )
        archive["mat_topology_indexes"][index] = int(
            observation.mat_topology_index
        )
        archive.flush()
        self._written_count = index + 1
        archive.attrs["written_count"] = self._written_count
        archive.flush()

    def _writer_loop(self):
        """在后台线程中独占HDF5句柄并处理队列。"""
        try:
            if self.resume_count == 0:
                self._create_file()
            else:
                self._validate_existing_file()
            with h5py.File(self.output_path, "r+") as archive:
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._STOP:
                            return
                        self._write_observation(archive, item)
                    finally:
                        self._queue.task_done()
        except Exception as exc:  # pragma: no cover - 由公共接口重新抛出
            self._error = exc
            # 文件创建或写入失败时必须释放队列计数，否则训练线程会永久阻塞。
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def _raise_background_error(self):
        """把后台线程异常重新抛到训练线程。"""
        if self._error is not None:
            raise RuntimeError("HDF5 探针后台写入失败。") from self._error

    def submit(self, observation):
        """提交一个完整探针时间点；队列满时提供有限背压。"""
        if self._closed:
            raise RuntimeError("HDF5 探针写入器已经关闭。")
        if not isinstance(observation, ProbeObservation):
            raise TypeError("observation 必须是 ProbeObservation。")
        self._raise_background_error()
        while True:
            try:
                self._queue.put(observation, timeout=0.2)
                break
            except queue.Full:
                self._raise_background_error()
                if not self._thread.is_alive():
                    raise RuntimeError("HDF5 探针后台线程意外退出。")
        self._raise_background_error()

    def flush(self):
        """等待全部已提交时间点落盘。"""
        while self._queue.unfinished_tasks:
            self._raise_background_error()
            if not self._thread.is_alive():
                raise RuntimeError("HDF5 探针后台线程意外退出。")
            threading.Event().wait(0.05)
        self._raise_background_error()

    def close(self):
        """提交停止标记、等待后台线程并传播写入异常。"""
        if self._closed:
            return
        self._raise_background_error()
        if self._thread.is_alive():
            while True:
                try:
                    self._queue.put(self._STOP, timeout=0.2)
                    break
                except queue.Full:
                    self._raise_background_error()
            self.flush()
        self._thread.join()
        self._closed = True
        self._raise_background_error()

    def __enter__(self):
        """返回上下文管理器中的写入器实例。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文时关闭后台写入器。"""
        self.close()
        return False
