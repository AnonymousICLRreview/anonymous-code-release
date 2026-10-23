"""Release regressions for data isolation, invalid outputs and link metrics."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ropc.datasets import (FeatureTransform, SingleGraphSource, collate_graphs,
                           load_dataset, rwse, _periodic_square_side)
from ropc.model import GraphData
from ropc.registry import dataset_spec
from ropc.tasks import (filtered_mrr, link_ranks, metric_from_predictions,
                        task_loss)


class DatasetEdgeCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_custom_value_fraction_and_invalid_fractions_reach_both_sources(self):
        for dataset in ("ogbn-arxiv", "Peptides-func"):
            with self.subTest(dataset=dataset):
                graph = load_dataset(dataset, "/unused", seed=7,
                                     synthetic=True, value_slice=.25)
                source = graph.source
                if source.is_family:
                    train = set(source.partitions["train"])
                    value = set(source.partitions["value"])
                    task = set(source.partitions["task"])
                    held_out = set().union(*(set(source.partitions[key]) for key in
                                             ("value", "tune", "audit", "reserve")))
                    self.assertEqual(train, task | held_out)
                    self.assertFalse(task & held_out)
                else:
                    train = set(graph.train_idx.tolist())
                    value = set(graph.value_idx.tolist())
                    task = set(graph.task_idx.tolist())
                    self.assertEqual(train, task | value)
                    self.assertFalse(task & value)
                self.assertEqual(len(value), max(1, round(.25 * len(train))))
                for fraction in (0., .5):
                    with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                        load_dataset(dataset, "/unused", synthetic=True,
                                     value_slice=fraction)

    def test_input_identity_excludes_random_controller_partition(self):
        graph = load_dataset("ogbn-arxiv", "/unused", synthetic=True, seed=9)
        first = SingleGraphSource(copy.copy(graph), dataset_spec(graph.name), seed=1,
                                  value_slice=.15)
        second = SingleGraphSource(copy.copy(graph), dataset_spec(graph.name), seed=2,
                                   value_slice=.25)
        self.assertEqual(first.manifest["input_sha256"], second.manifest["input_sha256"])
        self.assertNotEqual(first.manifest["tensor_sha256"], second.manifest["tensor_sha256"])
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_nonfinite_predictions_are_errors_even_where_labels_are_missing(self):
        metrics = ("accuracy", "balanced_accuracy", "macro_f1", "rocauc",
                   "ap", "mae", "log10_mse")
        for metric in metrics:
            target = (np.array([0, 1, -1]) if metric in metrics[:3]
                      else np.array([[0., 1.], [1., 0.], [np.nan, np.nan]]))
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(metric=metric, value=value):
                    pred = np.array([[2., 1.], [1., 2.], [0., 0.]])
                    pred[-1, 0] = value
                    with self.assertRaises(FloatingPointError):
                        metric_from_predictions(SimpleNamespace(metric=metric), pred, target)

        graph = load_dataset("ogbn-proteins", "/unused", synthetic=True)
        graph.y[0] = float("nan")
        H = torch.zeros(graph.num_nodes, graph.num_classes)
        H[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            task_loss(graph, H, SimpleNamespace(readout=torch.nn.Identity()))
        # Only selected supervised predictions contribute to the loss check.
        selected = task_loss(graph, H, SimpleNamespace(readout=torch.nn.Identity()),
                             torch.arange(1, graph.num_nodes))
        self.assertTrue(torch.isfinite(selected))

    def test_undefined_metrics_remain_undefined_and_all_missing_loss_is_zero(self):
        for metric in ("rocauc", "ap", "mae", "log10_mse"):
            with self.subTest(metric=metric):
                self.assertTrue(np.isnan(metric_from_predictions(
                    SimpleNamespace(metric=metric), np.zeros((4, 2)),
                    np.full((4, 2), np.nan))))
        for metric in ("rocauc", "ap"):
            self.assertTrue(np.isnan(metric_from_predictions(
                SimpleNamespace(metric=metric), np.zeros((4, 2)), np.ones((4, 2)))))

        for dataset in ("ogbn-arxiv", "ogbn-proteins", "LRIM-16"):
            with self.subTest(dataset=dataset):
                graph = load_dataset(dataset, "/unused", synthetic=True)
                graph.y.fill_(-1 if graph.task_type.endswith("classification") else float("nan"))
                H = torch.randn(graph.num_nodes, graph.num_classes, requires_grad=True)
                loss = task_loss(graph, H, SimpleNamespace(readout=torch.nn.Identity()))
                self.assertEqual(float(loss), 0.)
                loss.backward()
                self.assertTrue(torch.equal(H.grad, torch.zeros_like(H)))

    def test_loss_overflow_is_rejected_but_large_unlabeled_outputs_have_zero_loss(self):
        graph = load_dataset("LRIM-16", "/unused", synthetic=True)
        bb = SimpleNamespace(readout=torch.nn.Identity())
        H = torch.full((graph.num_nodes, 1), 1e20)
        with self.assertRaises(FloatingPointError):
            task_loss(graph, H, bb)
        graph.y.fill_(float("nan"))
        H = torch.full((graph.num_nodes, 1), torch.finfo(torch.float32).max,
                       requires_grad=True)
        loss = task_loss(graph, H, bb)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(H.grad, torch.zeros_like(H)))
        # Mask before squaring: a zero mask after an overflowing square can
        # still produce 0*inf = NaN during autograd for an unlabeled row.
        graph.y.fill_(0.)
        graph.y[0] = float("nan")
        H = torch.zeros(graph.num_nodes, 1)
        H[0] = torch.finfo(torch.float32).max
        H.requires_grad_()
        task_loss(graph, H, bb).backward()
        self.assertTrue(torch.equal(H.grad, torch.zeros_like(H)))

    @staticmethod
    def _link_graph(embeddings, positive_edges, sample_id=0):
        x = torch.tensor(embeddings, dtype=torch.float32).reshape(-1, 1)
        e = torch.tensor(positive_edges, dtype=torch.long).T.contiguous()
        labels = torch.ones(e.shape[1])
        indices = torch.arange(len(labels))
        return GraphData("PCQM-Contact", x, labels[:, None],
                         torch.empty(2, 0, dtype=torch.long), len(x), 1,
                         indices, indices, indices, task_type="link_prediction",
                         metric="filtered_mrr", edge_label_index=e,
                         edge_label=labels, sample_id=sample_id)

    def test_filtered_mrr_weights_graphs_equally_and_retains_self_candidates(self):
        # Filtering both known destinations leaves only a lower-scoring self
        # candidate in graph A; its two positive links each have rank one.
        a = self._link_graph([1., 2., 3.], [(0, 1), (0, 2)], sample_id=0)
        # The only positive in B has score two, behind candidates scoring three
        # and four. Its rank is three, irrespective of graph A's positive count.
        b = self._link_graph([1., 3., 2., 4.], [(0, 2)], sample_id=1)
        batch = collate_graphs([a, b])
        self.assertAlmostEqual(filtered_mrr(batch, batch.x), 2./3., places=7)
        self.assertNotAlmostEqual(filtered_mrr(batch, batch.x), 7./9., places=6)

        c = self._link_graph([1., .5, -1.], [(0, 1)])
        torch.testing.assert_close(link_ranks(c, c.x), torch.tensor([.5]))
        torch.testing.assert_close(link_ranks(c, c.x, exclude_self=True), torch.tensor([1.]))

    def test_link_metrics_reject_nonfinite_embeddings_and_overflowed_dot_products(self):
        graph = self._link_graph([1., .5, -1.], [(0, 1)])
        for value in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
            with self.subTest(value=value):
                H = torch.full_like(graph.x, value)
                for metric in (filtered_mrr, link_ranks):
                    with self.subTest(function=metric.__name__), self.assertRaises(FloatingPointError):
                        metric(graph, H)

    def test_prepared_feature_cache_cannot_claim_another_transform(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            self.skipTest("Optional PyG dataset dependencies are not installed")
        spec = dataset_spec("ZINC")
        raw = Data(x=torch.tensor([[1], [2]]),
                   edge_index=torch.tensor([[0, 1], [1, 0]]),
                   edge_attr=torch.tensor([0, 0]), y=torch.tensor([.1]))
        prepared = FeatureTransform(spec)(raw)
        self.assertEqual(prepared.ropc_feature_spec, repr(FeatureTransform(spec)))
        saved = prepared.x.clone()
        torch.testing.assert_close(FeatureTransform(spec)(prepared).x, saved)
        with self.assertRaises(ValueError):
            FeatureTransform(replace(spec, rwse_steps=8))(prepared.clone())
        legacy = prepared.clone()
        del legacy.ropc_feature_spec
        with self.assertRaisesRegex(ValueError, "(?i)fresh.*cache|fresh.*root"):
            FeatureTransform(spec)(legacy)

    def test_featureless_edgeless_graph_uses_explicit_node_count(self):
        try:
            from torch_geometric.data import Data
        except ImportError:
            self.skipTest("Optional PyG dataset dependencies are not installed")
        raw = Data(num_nodes=3, edge_index=torch.empty(2, 0, dtype=torch.long),
                   y=torch.tensor([0]))
        prepared = FeatureTransform(dataset_spec("MalNet-Tiny"))(raw)
        self.assertEqual(prepared.x.shape, (3, 5+16))
        self.assertTrue(torch.equal(prepared.x, torch.zeros_like(prepared.x)))

    @staticmethod
    def _periodic_edges(side):
        node = torch.arange(side * side)
        row, col = node // side, node % side
        dst = torch.stack((((row - 1) % side) * side + col,
                           ((row + 1) % side) * side + col,
                           row * side + (col - 1) % side,
                           row * side + (col + 1) % side), 1)
        return torch.stack((node.repeat_interleave(4), dst.flatten()))

    def test_exact_lattice_rwse_matches_sparse_powers_including_wraparound(self):
        for side in (3, 4, 16):
            with self.subTest(side=side):
                edge = self._periodic_edges(side)
                self.assertEqual(_periodic_square_side(*edge.numpy(), side*side), side)
                sparse = rwse(edge, side*side, 16)
                fast = rwse(edge, side*side, 16, cache_key=("lattice-test", side))
                torch.testing.assert_close(fast, sparse, rtol=0, atol=0)
                order = torch.randperm(edge.shape[1], generator=torch.Generator().manual_seed(14))
                shuffled = edge[:, order]
                self.assertEqual(_periodic_square_side(*shuffled.numpy(), side*side), side)
                torch.testing.assert_close(rwse(shuffled, side*side, 16, cache_key="shuffled-test"),
                                           sparse, rtol=0, atol=0)

    def test_lattice_cache_rejects_duplicate_directions_and_changed_topology(self):
        edge = self._periodic_edges(4)
        first = rwse(edge, 16, 8, cache_key="same-cache-request")
        changed = edge.clone()
        changed[1, 0] = changed[1, 1]  # same out-degree, wrong directional support
        self.assertIsNone(_periodic_square_side(*changed.numpy(), 16))
        expected = rwse(changed, 16, 8)
        actual = rwse(changed, 16, 8, cache_key="same-cache-request")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, actual))
        missing = edge[:, 1:]
        self.assertIsNone(_periodic_square_side(*missing.numpy(), 16))
        torch.testing.assert_close(rwse(missing, 16, 8, cache_key="same-cache-request"),
                                   rwse(missing, 16, 8), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
