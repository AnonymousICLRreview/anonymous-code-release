"""Official-cache compatibility checks with tiny local fixtures and no network."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data, InMemoryDataset
    from ogb.nodeproppred import NodePropPredDataset
    from ogb.graphproppred import PygGraphPropPredDataset
    from ropc.ogb_compat import load_ogb_node, load_ogb_graph
    HAVE_OGB = True
except ImportError:
    HAVE_OGB = False


@unittest.skipUnless(HAVE_OGB, "Install the declared OGB/PyG dataset dependencies")
class OGBCompatibilityTests(unittest.TestCase):
    def make_root(self, root, name, official_class, pyg_suffix=False):
        module = importlib.import_module(official_class.__module__)
        metadata = pd.read_csv(Path(module.__file__).with_name("master.csv"),
                               index_col=0, keep_default_na=False)[name]
        path = Path(root) / (name.replace("-", "_") + ("_pyg" if pyg_suffix else ""))
        (path / "processed").mkdir(parents=True)
        (path / f"RELEASE_v{metadata['version']}.txt").write_text("Official-format local test fixture\n")
        return path, metadata

    def test_node_cache_and_saved_split_use_explicit_pickle_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path, metadata = self.make_root(root, "ogbn-arxiv", NodePropPredDataset)
            graph = {"num_nodes": 3, "edge_index": np.array([[0, 1], [1, 2]]),
                     "node_feat": np.arange(6, dtype=np.float32).reshape(3, 2), "edge_feat": None}
            labels = np.array([[0], [1], [0]], dtype=np.int64)
            torch.save({"graph": graph, "labels": labels}, path / "processed" / "data_processed")
            split_path = path / "split" / metadata["split"]
            split_path.mkdir(parents=True)
            split = {"train": np.array([0]), "valid": np.array([1]), "test": np.array([2])}
            torch.save(split, split_path / "split_dict.pt")
            original_load = torch.load

            def checked_load(*args, **kwargs):
                self.assertIs(kwargs.get("weights_only"), False)
                self.assertEqual(kwargs.get("map_location"), "cpu")
                return original_load(*args, **kwargs)

            with patch("builtins.input", side_effect=AssertionError("No prompts allowed")), \
                 patch("ropc.ogb_compat.download_url", side_effect=AssertionError("No network allowed")), \
                 patch("torch.load", side_effect=checked_load):
                dataset = load_ogb_node("ogbn-arxiv", root)
                loaded_graph, loaded_labels = dataset[0]
                self.assertTrue(np.array_equal(loaded_labels, labels))
                self.assertTrue(np.array_equal(loaded_graph["node_feat"], graph["node_feat"]))
                self.assertTrue(np.array_equal(dataset.get_idx_split()["test"], np.array([2])))
                self.assertEqual(dataset.meta_info["eval metric"], metadata["eval metric"])
                self.assertEqual(dataset.processed_paths, [str(path / "processed" / "data_processed")])
                self.assertEqual(len(dataset), 1)

    def test_pyg_cache_retains_collation_splits_and_legacy_pyg_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path, metadata = self.make_root(root, "ogbg-molhiv", PygGraphPropPredDataset, pyg_suffix=True)
            graphs = [Data(x=torch.ones(2, 3), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([[0]])),
                      Data(x=torch.zeros(3, 3), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([[1]]))]
            data, slices = InMemoryDataset.collate(graphs)
            torch.save((data, slices), path / "processed" / "geometric_data_processed.pt")
            split_path = path / "split" / metadata["split"]
            split_path.mkdir(parents=True)
            for name, indices in {"train": [0], "valid": [1], "test": [1]}.items():
                pd.DataFrame(indices).to_csv(split_path / f"{name}.csv.gz", header=False, index=False, compression="gzip")
            with patch("builtins.input", side_effect=AssertionError("No prompts allowed")), \
                 patch("ropc.ogb_compat.download_url", side_effect=AssertionError("No network allowed")):
                dataset = load_ogb_graph("ogbg-molhiv", root)
                self.assertEqual(dataset.root, str(path))
                self.assertEqual(len(dataset), 2)
                self.assertEqual(dataset[1].num_nodes, 3)
                self.assertTrue(torch.equal(dataset[1].y, torch.tensor([[1]])))
                self.assertTrue(torch.equal(dataset.get_idx_split()["valid"], torch.tensor([1])))
                self.assertEqual(dataset.num_tasks, int(metadata["num tasks"]))

    def test_stale_release_is_rejected_without_deletion_or_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ogbn_arxiv"
            path.mkdir()
            existing = path / "existing.txt"
            existing.write_text("preserve this cache")
            with patch("builtins.input", side_effect=AssertionError("No prompts allowed")), \
                 patch("ropc.ogb_compat.download_url", side_effect=AssertionError("No network allowed")):
                with self.assertRaisesRegex(RuntimeError, "stale or incomplete"):
                    load_ogb_node("ogbn-arxiv", root)
            self.assertEqual(existing.read_text(), "preserve this cache")


if __name__ == "__main__":
    unittest.main()
