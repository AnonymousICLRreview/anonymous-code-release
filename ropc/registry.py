"""The 24 primary configurations in supplementary Table IX.

Dataset names, task semantics and batching live here. Experiment launchers only
override scientific hyperparameters; they never infer a task from label shape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str
    loader: str
    task_type: str
    metric: str
    metric_mode: str
    num_outputs: int
    L_star: int
    L_max: int
    bank: str
    r_min: float
    batch_size: int = 1
    subset_nodes: int = 0
    split_index: int = 0
    family: bool = True
    group: str = "development"
    rwse_steps: int = 16

    def config_overrides(self) -> dict:
        return {"dataset": self.name, "L_star": self.L_star,
                "L_max": self.L_max, "bank": self.bank, "r_min": self.r_min}

    def to_dict(self) -> dict:
        return asdict(self)


def _spec(name, loader, task, metric, outputs, depth, ceiling, bank, floor,
          batch=1, subset=0, family=True, group="development", rwse=16):
    return DatasetSpec(name, name.lower(), loader, task, metric,
                       "min" if metric in {"mae", "log10_mse"} else "max",
                       outputs, depth, ceiling, bank, floor, batch, subset,
                       family=family, group=group, rwse_steps=rwse)


_ROWS = [
    _spec("ogbn-arxiv", "ogb_node", "node_classification", "accuracy", 40,
          32, 64, "P0,Pr0,Pg,Pf", 2, family=False, rwse=0),
    _spec("ogbn-products", "ogb_node", "node_classification", "accuracy", 47,
          16, 64, "P0,Pg,Pf,Pc", 2, subset=50000, family=False, rwse=0),
    _spec("ogbn-proteins", "ogb_node", "node_multilabel", "rocauc", 112,
          16, 64, "P0,Pg,Pf,Px", 4, subset=50000, family=False, rwse=0),
    _spec("Roman-empire", "heterophily", "node_classification", "accuracy", 18,
          32, 64, "P0,Pr0,Pg,Px", 2, family=False, rwse=0),
    _spec("Amazon-ratings", "heterophily", "node_classification", "accuracy", 5,
          16, 64, "P0,Pg,Px,Pc", 2, family=False, rwse=0),
    _spec("Minesweeper", "heterophily", "node_binary", "rocauc", 1,
          16, 64, "P0,Pg,Pf,Pc", 4, family=False, rwse=0),
    _spec("PascalVOC-SP", "lrgb", "node_classification", "macro_f1", 21,
          64, 64, "P0,Pg,Pf,Pc", 4, batch=32),
    _spec("COCO-SP", "lrgb", "node_classification", "macro_f1", 81,
          64, 64, "P0,Pg,Pf,Pc", 4, batch=32),
    _spec("Peptides-func", "lrgb", "graph_multilabel", "ap", 10,
          32, 64, "P0,Pg,Pf,Px", 8, batch=128),
    _spec("Peptides-struct", "lrgb", "graph_regression", "mae", 11,
          16, 64, "P0,Pg,Pf,Px", 8, batch=128),
    _spec("PATTERN", "gnn_benchmark", "node_classification", "balanced_accuracy", 2,
          16, 64, "P0,Pg,Pc,Px", 2, batch=128),
    _spec("CLUSTER", "gnn_benchmark", "node_classification", "balanced_accuracy", 6,
          16, 64, "P0,Pg,Pc,Px", 2, batch=128),
    _spec("LRIM-16", "lrim", "node_regression", "log10_mse", 1,
          32, 32, "P0,Pg,Px,Pc", 8, batch=32),
    _spec("LRIM-32", "lrim", "node_regression", "log10_mse", 1,
          64, 64, "P0,Pg,Px,Pc", 8, batch=8),
    _spec("LRIM-64", "lrim", "node_regression", "log10_mse", 1,
          96, 128, "P0,Pg,Px,Pc", 8, batch=1),
    _spec("LRIM-128", "lrim", "node_regression", "log10_mse", 1,
          128, 128, "P0,Pg,Px,Pc", 8, batch=1),
    _spec("CIFAR10", "gnn_benchmark", "graph_classification", "accuracy", 10,
          32, 64, "P0,Pg,Pf,Pc", 4, batch=128, group="held-out"),
    _spec("ZINC", "zinc", "graph_regression", "mae", 1,
          16, 64, "P0,Pg,Pf,Px", 4, batch=128, group="held-out"),
    _spec("MNIST", "gnn_benchmark", "graph_classification", "accuracy", 10,
          32, 64, "P0,Pg,Pf,Pc", 4, batch=128, group="held-out"),
    _spec("MalNet-Tiny", "malnet", "graph_classification", "accuracy", 5,
          32, 64, "P0,Pr0,Pg,Pc", 4, batch=16, group="held-out"),
    _spec("ogbg-molhiv", "ogb_graph", "graph_binary", "rocauc", 1,
          16, 64, "P0,Pg,Pf,Px", 4, batch=128, group="held-out"),
    _spec("PCQM-Contact", "lrgb", "link_prediction", "filtered_mrr", 1,
          32, 64, "P0,Pg,Pf,Pc", 4, batch=128, group="held-out"),
    _spec("ogbg-molpcba", "ogb_graph", "graph_multilabel", "ap", 128,
          16, 64, "P0,Pg,Pf,Px", 4, batch=256, group="held-out"),
    _spec("ogbg-ppa", "ogb_graph", "graph_classification", "accuracy", 37,
          32, 64, "P0,Pg,Pf,Pc", 4, batch=64, group="held-out"),
]

DATASETS = {row.slug: row for row in _ROWS}
TRANSFER_DATASETS = {"lrim-256": _spec(
    "LRIM-256", "lrim", "node_regression", "log10_mse", 1,
    512, 64, "P0,Pg,Px,Pc", 8, batch=1, group="transfer")}


def dataset_spec(name_or_slug: str) -> DatasetSpec:
    key = name_or_slug.strip().lower().replace("_", "-")
    try:
        return {**DATASETS, **TRANSFER_DATASETS}[key]
    except KeyError:
        raise ValueError(f"Unknown dataset {name_or_slug!r}; choose from "
                         + ", ".join(DATASETS)) from None


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())
