"""Official benchmark adapters and disjoint controller-training partitions.

Downloads are lazy: importing ROPC or listing configurations never downloads data.
Only ``synthetic=True`` selects fixtures. A failed official download is an error,
never a reason to substitute a synthetic graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .model import GraphData
from .registry import DatasetSpec, dataset_spec

PREPROCESSING_VERSION = 1
ATOM_DIMS = (119, 4, 12, 12, 10, 6, 6, 2, 2)
BOND_DIMS = (5, 6, 2)
_RW_CACHE = {}


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _tensor(value, dtype=None):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype)


def _categorical(values, dims):
    values = values.long()
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] != len(dims):
        raise ValueError(f"Expected {len(dims)} categorical columns, got {values.shape[1]}")
    result = []
    for col, dim in enumerate(dims):
        v = values[:, col]
        if v.numel() and (int(v.min()) < 0 or int(v.max()) >= dim):
            raise ValueError(f"Categorical column {col} lies outside [0,{dim})")
        result.append(F.one_hot(v, dim).float())
    return torch.cat(result, -1)


def _edge_mean(edge_index, features, n):
    out = features.new_zeros(n, features.shape[1])
    out.index_add_(0, edge_index[1], features)
    degree = torch.bincount(edge_index[1], minlength=n).clamp_min(1)
    return out / degree.to(features.dtype).unsqueeze(-1)


def local_degree_profile(edge_index, n):
    src, dst = edge_index
    degree = torch.bincount(src, minlength=n).float()
    lo = torch.full((n,), torch.inf)
    hi = torch.zeros(n)
    lo.scatter_reduce_(0, src, degree[dst], reduce="amin", include_self=True)
    hi.scatter_reduce_(0, src, degree[dst], reduce="amax", include_self=True)
    total, square = torch.zeros(n), torch.zeros(n)
    total.index_add_(0, src, degree[dst])
    square.index_add_(0, src, degree[dst].square())
    mean = total / degree.clamp_min(1)
    std = (square / degree.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
    lo[~torch.isfinite(lo)] = 0
    return torch.stack((degree, lo, hi, mean, std), -1)


def _periodic_square_side(src, dst, n):
    """Recognize the canonical row-major, four-neighbor periodic lattice."""
    side = math.isqrt(n)
    if side < 3 or side * side != n or len(src) != 4 * n:
        return None
    if np.any(src < 0) or np.any(src >= n) or np.any(dst < 0) or np.any(dst >= n):
        return None
    row, col = src // side, src % side
    neighbors = (((row - 1) % side) * side + col,
                 ((row + 1) % side) * side + col,
                 row * side + (col - 1) % side,
                 row * side + (col + 1) % side)
    direction = np.full(len(src), -1, np.int64)
    for index, neighbor in enumerate(neighbors):
        direction[dst == neighbor] = index
    if np.any(direction < 0):
        return None
    # Exactly one arc in each direction; degree counts alone miss duplicates.
    if not np.all(np.bincount(4 * src + direction, minlength=4 * n) == 1):
        return None
    return side


def rwse(edge_index, n, steps=16, cache_key=None):
    """Exact return probabilities diag(P^k), k=1..steps, on input adjacency.

    This is input-only preprocessing, charged separately from forward latency.
    Sparse powers avoid an unconditional n-by-n dense allocation. A verified
    periodic LRIM lattice uses translation symmetry to evolve only one walk.
    Other families persist results via PyG's cache.
    """
    if steps <= 0:
        return torch.empty(n, 0)
    src, dst = edge_index.cpu().numpy()
    effective_key = None
    if cache_key is not None:
        side = _periodic_square_side(src, dst, n)
        if side is not None:
            effective_key = ("periodic_square", n, steps)
            if effective_key not in _RW_CACHE:
                walk = np.zeros((side, side), np.float64)
                walk[0, 0] = 1.
                returns = []
                for _ in range(steps):
                    walk = (np.roll(walk, 1, 0) + np.roll(walk, -1, 0)
                            + np.roll(walk, 1, 1) + np.roll(walk, -1, 1)) / 4.
                    returns.append(walk[0, 0])
                _RW_CACHE[effective_key] = torch.tensor(returns, dtype=torch.float32).repeat(n, 1)
            return _RW_CACHE[effective_key].clone()
        # Never reuse a size-only cache for a graph with a different topology.
        edge_hash = hashlib.sha256(np.ascontiguousarray(np.stack((src, dst))).tobytes()).hexdigest()
        effective_key = (cache_key, n, steps, edge_hash)
        if effective_key in _RW_CACHE:
            return _RW_CACHE[effective_key].clone()
    from scipy import sparse
    adjacency = sparse.csr_matrix((np.ones(len(src), np.float64), (src, dst)), shape=(n, n))
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(1)).ravel()
    transition = sparse.diags(1. / np.maximum(degree, 1.)) @ adjacency
    power = transition
    columns = []
    for step in range(steps):
        columns.append(power.diagonal())
        if step + 1 < steps:
            power = power @ transition
            power.eliminate_zeros()
    result = torch.from_numpy(np.stack(columns, 1).astype(np.float32))
    if effective_key is not None:
        _RW_CACHE[effective_key] = result
    return result.clone()


class FeatureTransform:
    """Explicit, label-free feature adapter, cached by official PyG datasets.

    Edge attributes enter the static encoder as mean incoming edge features.
    Molecular categories use fixed one-hot vocabularies; this is the ROPC input
    convention, not an assertion that the missing original encoder was recovered.
    """
    def __init__(self, spec):
        self.spec = spec

    def __repr__(self):
        return f"ROPCFeatures(v={PREPROCESSING_VERSION},dataset={self.spec.slug},rwse={self.spec.rwse_steps})"

    def __call__(self, data):
        if bool(_get(data, "ropc_prepared", False)):
            if _get(data, "ropc_feature_spec") != repr(self):
                raise ValueError("Cached ROPC preprocessing is missing its feature specification or uses "
                                 "a different specification. Select a fresh --data-root and prepare "
                                 "the dataset again.")
            return data
        spec = self.spec
        edge = _tensor(_get(data, "edge_index"), torch.long)
        if edge is None or edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        x = _tensor(_get(data, "x"))
        n = _get(data, "num_nodes")
        if n is None:
            n = len(x) if x is not None else (int(edge.max()) + 1 if edge.numel() else 0)
        n = int(n)
        edge_attr = _tensor(_get(data, "edge_attr"))
        molecule = spec.slug in {"peptides-func", "peptides-struct", "pcqm-contact", "ogbg-molhiv", "ogbg-molpcba"}
        if molecule:
            x = _categorical(x, ATOM_DIMS)
        elif spec.slug == "zinc":
            x = _categorical(x, (28,))
        elif spec.slug in {"pattern", "cluster"}:
            x = _categorical(x, (3 if spec.slug == "pattern" else 7,))
        elif spec.slug == "malnet-tiny":
            x = local_degree_profile(edge, n)
        elif x is None:
            x = torch.ones(n, 1)
        else:
            x = x.float().reshape(n, -1)
        pos = _tensor(_get(data, "pos"), torch.float32)
        if pos is not None and spec.slug in {"mnist", "cifar10"}:
            x = torch.cat((x, pos.reshape(n, -1)), -1)
        if edge_attr is not None:
            if molecule:
                ef = _categorical(edge_attr, BOND_DIMS)
            elif spec.slug == "zinc":
                ef = _categorical(edge_attr, (4,))
            else:
                ef = edge_attr.float().reshape(edge.shape[1], -1)
            x = torch.cat((x, _edge_mean(edge, ef, n)), -1)
        if spec.rwse_steps:
            key = (spec.slug, spec.rwse_steps) if spec.loader == "lrim" else None
            x = torch.cat((x, rwse(edge, n, spec.rwse_steps, key)), -1)
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite input features in {spec.name}")
        data.x = x.contiguous()
        data.ropc_prepared = True
        data.ropc_feature_spec = repr(self)
        return data


def select_mask(mask, split_index=0):
    mask = torch.as_tensor(mask)
    if mask.ndim == 2:
        if not 0 <= split_index < mask.shape[1]:
            raise ValueError(f"split_index must lie in [0,{mask.shape[1]})")
        mask = mask[:, split_index]
    if mask.ndim != 1:
        raise ValueError("Expected a 1D index/mask or [N,num_splits] mask")
    return torch.where(mask)[0] if mask.dtype == torch.bool else mask.long()


def _graph_from_data(data, spec, sample_id=None, source=None, split_index=0):
    x = _tensor(_get(data, "x"), torch.float32)
    edge = _tensor(_get(data, "edge_index"), torch.long)
    y = _tensor(_get(data, "y"))
    label_index = _tensor(_get(data, "edge_label_index", _get(data, "edge_index_labeled")), torch.long)
    label = _tensor(_get(data, "edge_label"), torch.float32)
    n = len(x)
    if spec.task_type == "link_prediction":
        if label is None or label_index is None:
            raise ValueError("PCQM-Contact release is missing labeled edge candidates")
        y = label.reshape(-1, 1)
    elif spec.task_type.endswith("classification"):
        y = y.reshape(-1).long()
    else:
        y = y.float().reshape(-1, spec.num_outputs)
    count = len(y)
    all_indices = torch.arange(count)
    split_indices = []
    for name in ("train", "val", "test"):
        mask = _get(data, name + "_mask")
        split_indices.append(select_mask(mask, split_index) if mask is not None else all_indices)
    graph = GraphData(spec.name, x, y, edge, n, spec.num_outputs, *split_indices,
                      task_type=spec.task_type, metric=spec.metric, metric_mode=spec.metric_mode,
                      edge_label_index=label_index, edge_label=label,
                      graph_ids=torch.tensor([sample_id]) if sample_id is not None else None,
                      source=source, sample_id=sample_id,
                      edge_attr=_tensor(_get(data, "edge_attr")))
    graph.metadata = {"dataset": spec.name, "preprocessing_version": PREPROCESSING_VERSION,
                      "rwse_steps": spec.rwse_steps, "split_index": split_index}
    return graph


def collate_graphs(graphs):
    """Disjoint-union batching preserving graph, node, and edge label scopes."""
    if not graphs:
        raise ValueError("Cannot collate an empty graph list")
    first = graphs[0]
    for graph in graphs:
        if (graph.name, graph.task_type, graph.num_classes, graph.x.shape[1]) != (
                first.name, first.task_type, first.num_classes, first.x.shape[1]):
            raise ValueError("A batch may contain only one compatible dataset schema")
    offsets = np.cumsum([0] + [g.num_nodes for g in graphs[:-1]])
    x = torch.cat([g.x for g in graphs])
    edges = torch.cat([g.edge_index + int(offset) for g, offset in zip(graphs, offsets)], 1)
    batch = torch.cat([torch.full((g.num_nodes,), i, dtype=torch.long) for i, g in enumerate(graphs)])
    y = torch.cat([g.y for g in graphs])
    indices = torch.arange(len(y))
    edge_label_index = (torch.cat([g.edge_label_index + int(o) for g, o in zip(graphs, offsets)], 1)
                        if first.edge_label_index is not None else None)
    edge_label = torch.cat([g.edge_label for g in graphs]) if first.edge_label is not None else None
    edge_attr = torch.cat([g.edge_attr for g in graphs]) if all(g.edge_attr is not None for g in graphs) else None
    result = GraphData(first.name, x, y, edges, len(x), first.num_classes,
                       indices, indices, indices, batch, len(graphs),
                       task_type=first.task_type, metric=first.metric, metric_mode=first.metric_mode,
                       edge_label_index=edge_label_index, edge_label=edge_label,
                       graph_ids=torch.tensor([g.sample_id for g in graphs]) if all(g.sample_id is not None for g in graphs) else None,
                       source=first.source, edge_attr=edge_attr)
    result.metadata = dict(getattr(first, "metadata", {}))
    return result


def _file_records(paths, cache_root):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "source_hashes.json"
    old = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    records = {}
    for path in sorted({str(Path(p).resolve()) for p in paths if Path(p).is_file()}):
        stat = Path(path).stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        previous = old.get(path, {})
        if previous.get("signature") == signature:
            digest = previous["sha256"]
        else:
            h = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        records[path] = {"signature": signature, "sha256": digest}
    temporary = cache_path.with_name(f"source_hashes.{__import__('uuid').uuid4().hex}.tmp")
    temporary.write_text(json.dumps(records, sort_keys=True, indent=2))
    temporary.replace(cache_path)
    return [{"name": Path(path).name, "bytes": rec["signature"][0], "sha256": rec["sha256"]}
            for path, rec in records.items()]


def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class DatasetSource:
    is_family = True

    def __init__(self, spec, split_datasets, seed=0, source_files=(), cache_root=None, synthetic=False, value_slice=.15):
        self.spec = spec
        self.name = spec.name
        self.batch_size = spec.batch_size
        self.synthetic = synthetic
        self.seed = seed
        self._datasets = split_datasets
        self._ranges = []
        self.partitions = {}
        offset = 0
        for split in ("train", "valid", "test"):
            dataset = split_datasets[split]
            self._ranges.append((offset, offset + len(dataset), dataset))
            self.partitions[split] = list(range(offset, offset + len(dataset)))
            offset += len(dataset)
        train_ids = np.asarray(self.partitions["train"])
        perm = np.random.default_rng(1000 + seed).permutation(train_ids)
        if not 0 < value_slice < .5:
            raise ValueError("value_slice must lie between zero and 0.5")
        sizes = [max(1, round(len(perm)*value_slice)), max(1, round(len(perm)*.10)),
                 min(2000, max(1, round(len(perm)*.10))), min(1600, max(1, round(len(perm)*.08)))]
        if sum(sizes) >= len(perm):
            raise ValueError("At least 6 training graphs are needed for disjoint ROPC partitions")
        pos = 0
        for split, size in zip(("value", "tune", "audit", "reserve"), sizes):
            self.partitions[split] = sorted(int(i) for i in perm[pos:pos+size])
            pos += size
        self.partitions["task"] = sorted(int(i) for i in perm[pos:])
        self.manifest = {"dataset": spec.name, "seed": seed, "synthetic": synthetic,
                         "task_type": spec.task_type, "preprocessing": repr(FeatureTransform(spec)),
                         "partition_sizes": {k: len(v) for k, v in self.partitions.items()},
                         "partition_sha256": _fingerprint(self.partitions)}
        if source_files:
            self.manifest["source_files"] = _file_records(source_files, cache_root)
            self.fingerprint_scope = "source_files_sha256_and_split_manifest"
        else:
            self.fingerprint_scope = "deterministic_fixture_spec_and_split_manifest"
            self.manifest["fixture_version"] = 1
        self.fingerprint = _fingerprint(self.manifest)

    def sample(self, sample_id: int, split=None):
        sample_id = int(sample_id)
        if split is not None and sample_id not in self.partitions[split]:
            raise ValueError(f"Graph {sample_id} is outside the {split} partition")
        for start, stop, data in self._ranges:
            if start <= sample_id < stop:
                raw = data[sample_id - start]
                if isinstance(raw, GraphData):
                    graph = copy.copy(raw)
                    graph.source = self
                    graph.sample_id = sample_id
                    graph.graph_ids = torch.tensor([sample_id])
                    graph.metadata = dict(getattr(graph, "metadata", {}))
                else:
                    raw = FeatureTransform(self.spec)(raw.clone())
                    graph = _graph_from_data(raw, self.spec, sample_id, self)
                graph.metadata.update({"fingerprint": self.fingerprint,
                                       "fingerprint_scope": self.fingerprint_scope,
                                       "synthetic": self.synthetic})
                return graph
        raise IndexError(f"Unknown graph id {sample_id} for {self.name}")

    def iter_batches(self, split, batch_size=None, epoch=0, shuffle=False) -> Iterator[GraphData]:
        ids = list(self.partitions[split])
        if shuffle:
            np.random.default_rng(self.seed + 65537*epoch).shuffle(ids)
        batch_size = int(batch_size or self.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(ids), batch_size):
            batch = [self.sample(i) for i in ids[start:start+batch_size]]
            yield batch[0] if len(batch) == 1 else collate_graphs(batch)


class SingleGraphSource:
    """Fixed global supervision split and 50k-node structural training blocks.

    Products/proteins use induced blocks of a reverse-Cuthill–McKee ordering.
    Each epoch shuffles block order. Validation/test always use the full graph.
    The particular partitioner was not supplied in the manuscript and is explicit
    here; it does not alter official labels or claim to recover the old sampler.
    """
    is_family = False

    def __init__(self, graph, spec, seed, subset_nodes=None, value_slice=.15):
        self.full_graph = copy.copy(graph)
        self.spec = spec
        self.name = spec.name
        self.subset_nodes = int(subset_nodes or spec.subset_nodes)
        self.batch_size = 1
        self.seed = seed
        input_sha256 = graph_fingerprint(graph, include_partition=False)
        train = graph.train_idx.cpu()
        if len(train) < 2:
            raise ValueError("Singlegraph training needs at least two labeled nodes")
        perm = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(1000+seed))]
        if not 0 < value_slice < .5:
            raise ValueError("value_slice must lie between zero and 0.5")
        nv = max(1, min(len(perm)-1, int(round(value_slice*len(perm)))))
        graph.value_idx = perm[:nv].sort().values
        graph.task_idx = perm[nv:].sort().values
        self.full_graph.value_idx = graph.value_idx
        self.full_graph.task_idx = graph.task_idx
        self._csr = None
        self._ordering = None
        self.manifest = {"dataset": spec.name, "seed": seed,
                         "input_sha256": input_sha256,
                         "tensor_sha256": graph_fingerprint(graph),
                         "subset_nodes": self.subset_nodes, "partitioner": "reverse_cuthill_mckee"}
        self.fingerprint_scope = "canonical_graph_tensors_and_supervision_splits"
        self.fingerprint = _fingerprint(self.manifest)

    def iter_training_batches(self, subset_nodes=None, epoch=0):
        from scipy import sparse
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        full = self.full_graph
        size = int(subset_nodes or self.subset_nodes or full.num_nodes)
        if size >= full.num_nodes:
            graph = copy.copy(full)
            graph.source = self
            yield graph
            return
        if self._csr is None:
            src, dst = full.edge_index.cpu().numpy()
            self._csr = sparse.csr_matrix((np.ones(len(src), dtype=np.bool_), (src, dst)),
                                          shape=(full.num_nodes, full.num_nodes))
            self._csr.sum_duplicates()
            self._ordering = reverse_cuthill_mckee(self._csr, symmetric_mode=False)
        blocks = list(range(math.ceil(full.num_nodes/size)))
        np.random.default_rng(self.seed+epoch).shuffle(blocks)
        for block in blocks:
            nodes_np = np.sort(self._ordering[block*size:(block+1)*size]).astype(np.int64)
            nodes = torch.from_numpy(nodes_np)
            local = self._csr[nodes_np][:, nodes_np].tocoo()
            edges = torch.from_numpy(np.stack((local.row, local.col)).astype(np.int64))
            mapping = torch.full((full.num_nodes,), -1, dtype=torch.long)
            mapping[nodes] = torch.arange(len(nodes))
            def remap(idx):
                result = mapping[idx.cpu()]
                return result[result >= 0]
            task, value = remap(full.task_idx), remap(full.value_idx)
            if not len(task) or not len(value):
                continue
            graph = GraphData(full.name, full.x[nodes], full.y[nodes], edges, len(nodes), full.num_classes,
                              remap(full.train_idx), remap(full.valid_idx), remap(full.test_idx),
                              task_type=full.task_type, metric=full.metric, metric_mode=full.metric_mode,
                              source=self, sample_id=block)
            graph.task_idx, graph.value_idx = task, value
            graph.metadata = {**getattr(full, "metadata", {}), "subgraph_id": block,
                              "global_node_ids_sha256": hashlib.sha256(nodes_np.tobytes()).hexdigest()}
            yield graph


def graph_fingerprint(graph, include_partition=True):
    h = hashlib.sha256()
    for name in ("x", "edge_index", "y", "train_idx", "valid_idx", "test_idx", "task_idx", "value_idx",
                 "edge_label_index", "edge_label"):
        value = getattr(graph, name, None)
        if not include_partition and name in {"task_idx", "value_idx"}:
            continue
        if value is not None:
            value = value.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            array = value.numpy()
            h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _pyg():
    try:
        import torch_geometric.datasets as datasets
    except ImportError as exc:
        raise RuntimeError("Install dataset dependencies: pip install -e '.[datasets]'") from exc
    return datasets


def _official_single(spec, root, seed, split_index, value_slice=.15):
    if spec.loader == "ogb_node":
        try:
            from .ogb_compat import load_ogb_node
        except ImportError as exc:
            raise RuntimeError("Install OGB: pip install -e '.[datasets]'") from exc
        dataset = load_ogb_node(name=spec.name, root=str(root))
        raw, labels = dataset[0]
        splits = dataset.get_idx_split()
        edge = _tensor(raw["edge_index"], torch.long)
        n = int(raw["num_nodes"])
        x = _tensor(raw.get("node_feat"), torch.float32)
        if x is None:
            ef = _tensor(raw.get("edge_feat"), torch.float32)
            if ef is None:
                raise ValueError(f"{spec.name} has neither node nor edge features")
            x = _edge_mean(edge, ef, n)
        y = torch.as_tensor(labels)
        y = y.reshape(-1).long() if spec.task_type.endswith("classification") else y.float().reshape(n, -1)
        graph = GraphData(spec.name, x, y, edge, n, spec.num_outputs,
                          *[torch.as_tensor(splits[s], dtype=torch.long) for s in ("train", "valid", "test")],
                          task_type=spec.task_type, metric=spec.metric, metric_mode=spec.metric_mode)
    else:
        dataset = _pyg().HeterophilousGraphDataset(root=str(root), name=spec.name)
        graph = _graph_from_data(dataset[0], spec, split_index=split_index)
    graph.metadata = {"dataset": spec.name, "split_index": split_index, "synthetic": False,
                      "preprocessing_version": PREPROCESSING_VERSION}
    graph.source = SingleGraphSource(graph, spec, seed, value_slice=value_slice)
    graph.metadata.update({"fingerprint": graph.source.fingerprint,
                           "fingerprint_scope": graph.source.fingerprint_scope})
    return graph


class _LRIMSequence:
    def __init__(self, data, start, stop, spec):
        self.data, self.start, self.stop, self.spec = data, start, stop, spec
    def __len__(self):
        return self.stop - self.start
    def __getitem__(self, index):
        data = self.data[self.start+index].clone()
        edge = data.edge_index.long()
        n = len(data.x)
        keys = torch.cat((edge[0]*n+edge[1], edge[1]*n+edge[0])).unique(sorted=True)
        data.edge_index = torch.stack((keys//n, keys % n))
        data.x, data.y = data.x.float(), data.y.float()
        return FeatureTransform(self.spec)(data)


def _official_family(spec, root, seed, value_slice=.15):
    files = []
    transform = FeatureTransform(spec)
    if spec.loader == "lrim":
        _pyg()  # official release contains PyG Data objects
        from huggingface_hub import hf_hub_download
        side = int(spec.slug.split("-")[-1])
        filename = f"lrim_{side}_0.6_10k.pt"
        raw_dir = root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / filename
        if not path.exists():
            path = Path(hf_hub_download(repo_id="jmathys/lrim_graph_benchmark", filename=filename,
                                       repo_type="dataset", local_dir=str(raw_dir)))
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if len(data) != 10000:
            raise ValueError(f"Official LRIM release must contain 10,000 graphs; got {len(data)}")
        split_data = {"train": _LRIMSequence(data, 0, 8000, spec),
                      "valid": _LRIMSequence(data, 8000, 9000, spec),
                      "test": _LRIMSequence(data, 9000, 10000, spec)}
        files.append(path)
    elif spec.loader == "ogb_graph":
        from .ogb_compat import load_ogb_graph
        data = load_ogb_graph(name=spec.name, root=str(root), pre_transform=transform)
        split = data.get_idx_split()
        split_data = {name: data[split[name]] for name in ("train", "valid", "test")}
        files.extend(data.processed_paths)
        split_file = Path(data.root) / "split" / data.meta_info["split"] / "split_dict.pt"
        if split_file.exists():
            files.append(split_file)
        else:
            files.extend(split_file.parent / (split + ".csv.gz") for split in ("train", "valid", "test"))
    else:
        api = _pyg()
        split_data = {}
        for official, name in (("train", "train"), ("val", "valid"), ("test", "test")):
            kwargs = dict(root=str(root), split=official, pre_transform=transform)
            if spec.loader == "lrgb":
                data = api.LRGBDataset(name=spec.name, **kwargs)
            elif spec.loader == "gnn_benchmark":
                data = api.GNNBenchmarkDataset(name=spec.name, **kwargs)
            elif spec.loader == "zinc":
                data = api.ZINC(subset=True, **kwargs)
            elif spec.loader == "malnet":
                data = api.MalNetTiny(**kwargs)
            else:
                raise ValueError(f"Unsupported family loader {spec.loader}")
            split_data[name] = data
            files.extend(data.processed_paths)
    source = DatasetSource(spec, split_data, seed, files, root / "ropc_manifest", value_slice=value_slice)
    return source.sample(source.partitions["task"][0])


def synthetic_fixture(spec, seed=0, family_graphs=32, nodes=24, value_slice=.15):
    """Small deterministic interface fixtures, never benchmark measurements."""
    from types import SimpleNamespace
    def make(sample_id):
        rng = torch.Generator().manual_seed(seed*100003 + sample_id + 17)
        n = nodes + sample_id % 3
        x = torch.randn(n, 8, generator=rng)
        row = torch.arange(n)
        edge = torch.stack((torch.cat((row, row)), torch.cat(((row+1) % n, (row-1) % n))))
        scope = 1 if spec.task_type.startswith("graph_") else n
        kind = spec.task_type
        if kind.endswith("classification"):
            y = (torch.arange(scope)+sample_id) % spec.num_outputs
        elif "regression" in kind:
            base = x[:, :1] if scope == n else x.mean(0, keepdim=True)[:, :1]
            y = base.repeat(1, spec.num_outputs) + torch.arange(spec.num_outputs)[None, :]/10
        else:
            y = (torch.arange(scope)[:, None]+torch.arange(spec.num_outputs)[None, :]+sample_id).remainder(2).float()
            if "multilabel" in kind and scope > 1:
                y[0, 0] = float("nan")
        label_index = label = None
        if kind == "link_prediction":
            label_index = torch.stack((torch.arange(n), (torch.arange(n)+2) % n))
            label = torch.arange(n).remainder(2).float()
            y = label[:, None]
        count = len(y)
        if spec.family:
            train = valid = test = torch.arange(count)
        else:
            train = torch.arange(0, max(4, int(.6*count)))
            valid = torch.arange(max(4, int(.6*count)), int(.8*count))
            test = torch.arange(int(.8*count), count)
        graph = GraphData(spec.name, x, y, edge, n, spec.num_outputs, train, valid, test,
                          task_type=kind, metric=spec.metric, metric_mode=spec.metric_mode,
                          edge_label_index=label_index, edge_label=label,
                          sample_id=sample_id)
        graph.metadata = {"synthetic": True, "fixture_seed": seed, "fixture_version": 1}
        return graph
    if not spec.family:
        graph = make(0)
        graph.source = SingleGraphSource(graph, spec, seed, subset_nodes=spec.subset_nodes, value_slice=value_slice)
        graph.metadata["fingerprint"] = graph.source.fingerprint
        graph.metadata["fingerprint_scope"] = graph.source.fingerprint_scope
        return graph
    split_data = {"train": [make(i) for i in range(family_graphs)],
                  "valid": [make(i+family_graphs) for i in range(8)],
                  "test": [make(i+family_graphs+8) for i in range(8)]}
    source = DatasetSource(spec, split_data, seed, synthetic=True, value_slice=value_slice)
    return source.sample(source.partitions["task"][0])


def load_dataset(name, root, seed=0, split_index=0, synthetic=False, value_slice=.15):
    spec = dataset_spec(name)
    if synthetic:
        return synthetic_fixture(spec, seed, value_slice=value_slice)
    root = Path(root).expanduser().resolve() / spec.slug
    root.mkdir(parents=True, exist_ok=True)
    # Shared caches are mutated by first-time downloads/preprocessing. Slurm
    # seeds may start together, so serialize those constructors on this dataset.
    import fcntl
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return (_official_family(spec, root, seed, value_slice) if spec.family else
                _official_single(spec, root, seed, split_index, value_slice))
