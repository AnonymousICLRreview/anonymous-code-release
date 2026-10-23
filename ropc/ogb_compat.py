"""Narrow OGB cache/batch compatibility without global torch.load overrides.

OGB 1.3.6 uses pickle-backed official caches and interactive download/version
questions. These adapters retain its metadata, readers, processors and splits,
while loading only its named caches with explicit weights_only=False. Dataset
roots must contain trusted official OGB data. Stale releases are never deleted.
An explicit prepare/run invocation authorizes the noninteractive download.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import tempfile
import zipfile

import pandas as pd
import torch
from ogb.nodeproppred import NodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.utils.url import download_url, extract_zip
from torch_geometric.data import InMemoryDataset


def _configure(dataset, name, root, official_class, graph=False):
    module = importlib.import_module(official_class.__module__)
    master = pd.read_csv(Path(module.__file__).with_name("master.csv"),
                         index_col=0, keep_default_na=False)
    if name not in master:
        raise ValueError(f"Unknown official OGB dataset {name!r}")
    dataset.name, dataset.meta_info = name, master[name]
    dataset.original_root = str(Path(root).expanduser().resolve())
    dirname = name.replace("-", "_")
    if graph and (Path(dataset.original_root) / (dirname + "_pyg")).exists():
        dirname += "_pyg"
    dataset.dir_name = dirname
    dataset.root = str(Path(dataset.original_root) / dirname)
    release = Path(dataset.root) / f"RELEASE_v{dataset.meta_info['version']}.txt"
    if Path(dataset.root).exists() and not release.is_file():
        raise RuntimeError(f"OGB cache {dataset.root} lacks {release.name}. It is stale or incomplete. "
                           "Use a fresh dataset root or restore the matching official release; no files were deleted.")
    dataset.download_name = dataset.meta_info["download_name"]
    dataset.num_tasks = int(dataset.meta_info["num tasks"])
    dataset.task_type = dataset.meta_info["task type"]
    dataset.eval_metric = dataset.meta_info["eval metric"]
    dataset.binary = dataset.meta_info["binary"] == "True"
    dataset.is_hetero = dataset.meta_info.get("is hetero", "False") == "True"
    if graph:
        dataset.__num_classes__ = int(dataset.meta_info["num classes"])
    else:
        dataset.num_classes = int(dataset.meta_info["num classes"])


class _CacheAndDownload:
    def get_idx_split(self, split_type=None):
        split_type = self.meta_info["split"] if split_type is None else split_type
        path = Path(self.root) / "split" / split_type / "split_dict.pt"
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)
        return super().get_idx_split(split_type)

    def download(self):
        destination = Path(self.root)
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"Incomplete OGB cache at {destination}; use a fresh root. Existing files were preserved.")
        Path(self.original_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ropc-ogb-", dir=self.original_root) as temporary:
            archive = download_url(self.meta_info["url"], temporary)
            # Preserve OGB's extractor, after rejecting paths outside staging.
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if not (Path(temporary) / name).resolve().is_relative_to(Path(temporary).resolve()):
                        raise ValueError("OGB archive contains an unsafe path")
            extract_zip(archive, temporary)
            source = Path(temporary) / self.download_name
            release = source / f"RELEASE_v{self.meta_info['version']}.txt"
            if not source.is_dir() or not release.is_file():
                raise RuntimeError("Downloaded OGB archive does not match its official release metadata")
            if destination.exists():
                destination.rmdir()  # only an empty directory can reach this line
            shutil.move(str(source), str(destination))


class OfficialNodeDataset(_CacheAndDownload, NodePropPredDataset):
    def __init__(self, name, root):
        _configure(self, name, root, NodePropPredDataset)
        self.pre_process()

    @property
    def processed_paths(self):
        return [str(Path(self.root) / "processed" / "data_processed")]

    def pre_process(self):
        cache = Path(self.processed_paths[0])
        if cache.is_file():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            self.graph, self.labels = stored["graph"], stored["labels"]
            return
        filename = ("edge_index_dict.npz" if self.is_hetero else "data.npz") if self.binary else (
            "triplet-type-list.csv.gz" if self.is_hetero else "edge.csv.gz")
        if not (Path(self.root) / "raw" / filename).is_file():
            self.download()
        cache.parent.mkdir(parents=True, exist_ok=True)
        # With raw files present and no cache, upstream uses its official raw
        # readers and creates the cache; neither interactive branch is entered.
        super().pre_process()


class OfficialGraphDataset(_CacheAndDownload, PygGraphPropPredDataset):
    def __init__(self, name, root, pre_transform=None):
        _configure(self, name, root, PygGraphPropPredDataset, graph=True)
        # Skip OGB's interactive constructor and its implicit torch.load default;
        # retain PyG lifecycle and OGB's process()/raw_file_names implementation.
        InMemoryDataset.__init__(self, self.root, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location="cpu", weights_only=False)

    def _download(self):
        if Path(self.processed_paths[0]).is_file() or all(Path(p).is_file() for p in self.raw_paths):
            return
        self.download()


def load_ogb_node(name, root):
    """Load an official node-property dataset with batch-safe cache handling."""
    return OfficialNodeDataset(name, root)


def load_ogb_graph(name, root, pre_transform=None):
    """Load an official PyG graph-property dataset with its original processor."""
    return OfficialGraphDataset(name, root, pre_transform=pre_transform)
