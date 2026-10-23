"""Tests for dataset isolation, readout scope, official metrics and gradients."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ropc.datasets import (FeatureTransform, SingleGraphSource, collate_graphs,
                           load_dataset, rwse, select_mask)
from ropc.registry import DATASETS, dataset_spec
from ropc.tasks import (filtered_mrr, metric_from_predictions, node_uncertainty,
                        predictions, task_loss, targets_for)


class DatasetTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_twenty_four_task_shapes_and_gradients(self):
        self.assertEqual(len(DATASETS), 24)
        for spec in DATASETS.values():
            with self.subTest(dataset=spec.name):
                graph = load_dataset(spec.name, "/unused", seed=3, synthetic=True)
                if spec.family:
                    graph = next(graph.source.iter_batches("task", 2))
                H = torch.randn(graph.num_nodes, 12, requires_grad=True)
                bb = SimpleNamespace(readout=torch.nn.Linear(12, spec.num_outputs))
                loss = task_loss(graph, H, bb)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(H.grad).all())
                pred = predictions(graph, H, bb)
                self.assertEqual(pred.shape[0], len(targets_for(graph)))
                self.assertEqual(pred.shape[-1], spec.num_outputs)
                uncertainty = node_uncertainty(graph, H, bb)
                self.assertEqual(uncertainty.shape, (graph.num_nodes,))
                self.assertTrue(torch.isfinite(uncertainty).all())

    def test_family_graph_partitions_are_disjoint_and_reproducible(self):
        a = load_dataset("Peptides-func", "/unused", seed=5, synthetic=True).source
        b = load_dataset("Peptides-func", "/unused", seed=5, synthetic=True).source
        c = load_dataset("Peptides-func", "/unused", seed=6, synthetic=True).source
        self.assertEqual(a.partitions, b.partitions)
        self.assertNotEqual(a.partitions["task"], c.partitions["task"])
        labels = ("task", "value", "tune", "audit", "reserve", "valid", "test")
        for i, key in enumerate(labels):
            for other in labels[i+1:]:
                self.assertFalse(set(a.partitions[key]) & set(a.partitions[other]))
        self.assertEqual(set().union(*(set(a.partitions[k]) for k in labels[:5])),
                         set(a.partitions["train"]))
        with self.assertRaises(ValueError):
            a.sample(a.partitions["test"][0], split="task")

    def test_disjoint_batching_does_not_mix_graph_readouts(self):
        source = load_dataset("ZINC", "/unused", synthetic=True).source
        graphs = [source.sample(i) for i in source.partitions["task"][:3]]
        merged = collate_graphs(graphs)
        bb = SimpleNamespace(readout=torch.nn.Linear(8, 1))
        single = torch.cat([predictions(g, g.x, bb) for g in graphs])
        together = predictions(merged, merged.x, bb)
        torch.testing.assert_close(single, together)
        self.assertTrue(torch.equal(merged.batch_vec[merged.edge_index[0]],
                                    merged.batch_vec[merged.edge_index[1]]))

    def test_split_columns_are_selected_not_flattened(self):
        masks = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
        self.assertEqual(select_mask(masks, 0).tolist(), [0])
        self.assertEqual(select_mask(masks, 1).tolist(), [1])
        with self.assertRaises(ValueError):
            select_mask(masks, 2)

    def test_subgraph_supervision_preserves_global_value_holdout(self):
        graph = load_dataset("ogbn-products", "/unused", seed=0, synthetic=True)
        graph.x = torch.arange(graph.num_nodes).float()[:, None]
        source = SingleGraphSource(graph, dataset_spec(graph.name), 0, subset_nodes=12)
        global_task, global_value = set(graph.task_idx.tolist()), set(graph.value_idx.tolist())
        seen = 0
        for sub in source.iter_training_batches(epoch=0):
            task_ids = set(sub.x[sub.task_idx, 0].long().tolist())
            value_ids = set(sub.x[sub.value_idx, 0].long().tolist())
            self.assertTrue(task_ids <= global_task)
            self.assertTrue(value_ids <= global_value)
            self.assertFalse(task_ids & global_value)
            self.assertLessEqual(sub.num_nodes, 12)
            seen += 1
        self.assertGreater(seen, 0)

    def test_missing_multilabel_targets_do_not_create_gradients(self):
        graph = load_dataset("ogbn-proteins", "/unused", synthetic=True)
        graph.y[:] = float("nan")
        graph.y[1, 0], graph.y[2, 0] = 0., 1.
        H = torch.randn(graph.num_nodes, 112, requires_grad=True)
        bb = SimpleNamespace(readout=torch.nn.Identity())
        loss = task_loss(graph, H, bb)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(H.grad)), 2)
        per_item = task_loss(graph, H, bb, reduction="none")
        self.assertTrue(torch.isnan(per_item[0]))
        self.assertTrue(torch.isfinite(per_item[1:3]).all())

    def test_auc_ap_equal_official_sklearn_with_ties_and_missing_labels(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = np.array([[0, 1, np.nan], [1, 0, np.nan], [1, 1, np.nan], [0, np.nan, np.nan]])
        p = np.array([[.2, .9, 0], [.8, .2, 0], [.8, .2, 0], [.2, .1, 0]])
        for name, reference in (("rocauc", roc_auc_score), ("ap", average_precision_score)):
            expected = []
            for col in (0, 1):
                mask = np.isfinite(y[:, col])
                expected.append(reference(y[mask, col], p[mask, col]))
            actual = metric_from_predictions(SimpleNamespace(metric=name), p, y)
            self.assertAlmostEqual(actual, np.mean(expected), places=12)

    def test_macro_f1_and_log_mse_are_not_accuracy_or_absolute_gap(self):
        from sklearn.metrics import f1_score
        y = np.array([0, 0, 1, 2])
        p = np.array([[3, 1, 0], [1, 3, 0], [1, 3, 0], [1, 2, 4]])
        got = metric_from_predictions(SimpleNamespace(metric="macro_f1"), p, y)
        self.assertAlmostEqual(got, f1_score(y, p.argmax(1), average="macro"))
        got = metric_from_predictions(SimpleNamespace(metric="log10_mse"), np.array([[.01], [.01]]), np.zeros((2, 1)))
        self.assertAlmostEqual(got, -4.)

    def test_filtered_mrr_matches_released_lrgb_evaluator(self):
        source = load_dataset("PCQM-Contact", "/unused", synthetic=True).source
        gs = [source.sample(i) for i in source.partitions["test"][:2]]
        batch = collate_graphs(gs)
        H = torch.randn(batch.num_nodes, 7, generator=torch.Generator().manual_seed(4))
        expected = []
        offset = 0
        for g in gs:
            h = H[offset:offset+g.num_nodes]
            score = h @ h.T
            e = g.edge_label_index[:, g.edge_label == 1]
            pos = score[e[0], e[1]].clone()
            score[e[0], e[1]] = -torch.inf
            order = torch.argsort(torch.cat([pos[:, None], score[e[0]]], 1), 1, descending=True)
            rank = torch.nonzero(order == 0)[:, 1]+1
            expected.append(float((1./rank).mean()))
            offset += g.num_nodes
        self.assertAlmostEqual(filtered_mrr(batch, H), np.mean(expected), places=7)

    def test_rwse_matches_dense_markov_chain(self):
        edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        transition = np.array([[0, 1, 0], [.5, 0, .5], [0, 1, 0]])
        actual = rwse(edge, 3, 5).numpy()
        expected = np.stack([np.diag(np.linalg.matrix_power(transition, k)) for k in range(1, 6)], 1)
        np.testing.assert_allclose(actual, expected)

    def test_feature_adapter_uses_edge_attributes_without_labels(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            self.skipTest("Optional official dataset dependencies are not installed")
        spec = dataset_spec("ZINC")
        a = Data(x=torch.tensor([[1], [2], [3]]), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                 edge_attr=torch.tensor([1, 1, 2, 2]), y=torch.tensor([.3]))
        b = a.clone()
        b.y[:] = 1000.
        fa, fb = FeatureTransform(spec)(a), FeatureTransform(spec)(b)
        torch.testing.assert_close(fa.x, fb.x)
        self.assertEqual(fa.x.shape, (3, 28+4+16))


if __name__ == "__main__":
    unittest.main()
