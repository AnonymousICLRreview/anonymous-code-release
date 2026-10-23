"""Task-specific readouts, masked losses, and benchmark metrics.

Metrics are fractions (not percentages). AUC/AP aggregate predictions over the
whole evaluation split. PCQM-Contact follows LRGB's graph-averaged filtered MRR.
No evaluation function fits a parameter or reads training labels.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def targets_for(graph):
    return graph.edge_label if graph.task_type == "link_prediction" else graph.y


def predictions(graph, H, bb):
    kind = graph.task_type
    if kind == "link_prediction":
        e = graph.edge_label_index
        if e is None:
            raise ValueError("Link prediction requires edge_label_index")
        return (H[e[0]] * H[e[1]]).sum(-1, keepdim=True)
    if kind.startswith("graph_"):
        pooled = H.new_zeros(graph.num_graphs, H.shape[-1])
        pooled.index_add_(0, graph.batch_vec, H)
        counts = torch.bincount(graph.batch_vec, minlength=graph.num_graphs)
        pooled = pooled / counts.clamp_min(1).to(H.dtype).unsqueeze(-1)
        return bb.readout(pooled)
    return bb.readout(H)


def item_batch(graph):
    if graph.task_type.startswith("graph_"):
        return torch.arange(graph.num_graphs, device=graph.x.device)
    if graph.task_type == "link_prediction":
        return graph.batch_vec[graph.edge_label_index[0]]
    return graph.batch_vec


def task_loss(graph, H, bb, indices=None, reduction="mean"):
    pred = predictions(graph, H, bb)
    target = targets_for(graph).to(pred.device)
    if indices is not None:
        indices = indices.to(pred.device)
        pred, target = pred[indices], target[indices]
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Non-finite task predictions; the loss cannot discard failed predictions")
    kind = graph.task_type
    if kind.endswith("classification"):
        target = target.reshape(-1).long()
        valid = (target >= 0) & (target < pred.shape[-1])
        safe = torch.where(valid, target, torch.zeros_like(target))
        weight = None
        if graph.name in {"PATTERN", "CLUSTER", "PascalVOC-SP", "COCO-SP"}:
            counts = torch.bincount(target[valid], minlength=pred.shape[-1]).to(pred.dtype)
            weight = (valid.sum() - counts) / valid.sum().clamp_min(1)
            weight = torch.where(counts > 0, weight, torch.zeros_like(weight))
            # Equal labels in a tiny fixture should not erase the training loss.
            if float(weight.sum()) == 0:
                weight = torch.ones_like(weight)
            norm = weight[safe[valid]].mean() if valid.any() else pred.new_tensor(1.)
            weight = weight / norm.clamp_min(1e-12)
        losses = F.cross_entropy(pred, safe, weight=weight, reduction="none")
        losses = torch.where(valid, losses, torch.full_like(losses, float("nan")))
        valid_items = valid
    else:
        target = target.to(pred.dtype).reshape_as(pred)
        valid = torch.isfinite(target)
        safe = torch.where(valid, target, torch.zeros_like(target))
        safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
        if "regression" in kind:
            losses = ((safe_pred - safe).square() if graph.metric == "log10_mse"
                      else (safe_pred - safe).abs())
        else:
            losses = F.binary_cross_entropy_with_logits(safe_pred, safe, reduction="none")
        losses = torch.where(valid, losses, torch.zeros_like(losses))
        count = valid.sum(-1)
        losses = losses.sum(-1) / count.clamp_min(1)
        losses = torch.where(count > 0, losses, torch.full_like(losses, float("nan")))
        valid_items = count > 0
    if not torch.isfinite(losses[valid_items]).all():
        raise FloatingPointError("Non-finite task loss for a labeled item")
    if reduction == "none":
        return losses
    finite = torch.isfinite(losses)
    if reduction == "sum":
        return losses[finite].sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported loss reduction {reduction!r}")
    return losses[finite].mean() if finite.any() else (pred * 0.).sum()


def node_uncertainty(graph, H, bb):
    """Label-free sensor channel; see docs/DATASETS.md for regression convention."""
    if graph.task_type == "link_prediction":
        src, dst = graph.edge_index
        score = (H[src] * H[dst]).sum(-1)
        p = score.sigmoid().clamp(1e-7, 1-1e-7)
        entropy = -(p * p.log() + (1-p) * (1-p).log()) / math.log(2)
        out = H.new_zeros(graph.num_nodes)
        out.index_add_(0, src, entropy)
        return out / torch.bincount(src, minlength=graph.num_nodes).clamp_min(1)
    pred = bb.readout(H)
    if graph.task_type.endswith("classification"):
        p = pred.softmax(-1).clamp_min(1e-8)
        return -(p*p.log()).sum(-1) / math.log(max(2, pred.shape[-1]))
    if "regression" in graph.task_type:
        means = pred.new_zeros(graph.num_graphs, pred.shape[-1])
        means.index_add_(0, graph.batch_vec, pred)
        means /= torch.bincount(graph.batch_vec, minlength=graph.num_graphs).clamp_min(1).unsqueeze(-1)
        return torch.log1p((pred - means[graph.batch_vec]).square().mean(-1).sqrt())
    p = pred.sigmoid().clamp(1e-7, 1-1e-7)
    return -(p*p.log() + (1-p)*(1-p).log()).mean(-1) / math.log(2)


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _auc(y, p):
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    positive_counts = np.add.reduceat(sorted_y, starts)
    rank_sum = float(np.dot(positive_counts, (starts + 1 + ends) / 2))
    positives = float(y.sum())
    return (rank_sum - positives*(positives+1)/2) / (positives*(len(y)-positives))


def _ap(y, p):
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p)-1]
    true_positive = np.cumsum(sorted_y)[ends]
    recall = true_positive / y.sum()
    precision = true_positive / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def metric_from_predictions(graph_or_task_spec, predictions, targets):
    """Exact split-level metric; ROC-AUC/AP skip tasks lacking either class."""
    metric = graph_or_task_spec.metric
    p, y = _array(predictions), _array(targets)
    if len(y) == 0:
        return float("nan")
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite predictions cannot contribute to a benchmark metric")
    if metric in {"accuracy", "balanced_accuracy", "macro_f1"}:
        truth = y.reshape(-1).astype(np.int64)
        guess = p.argmax(-1) if p.ndim > 1 and p.shape[-1] > 1 else (p.reshape(-1) > 0).astype(np.int64)
        valid = truth >= 0
        truth, guess = truth[valid], guess[valid]
        if not len(truth):
            return float("nan")
        if metric == "accuracy":
            return float(np.mean(truth == guess))
        classes = np.union1d(truth, guess)
        scores = []
        for cls in classes:
            actual, predicted = truth == cls, guess == cls
            tp = np.sum(actual & predicted)
            if metric == "balanced_accuracy":
                scores.append(tp / actual.sum() if actual.any() else 0.)
            else:
                scores.append(2*tp / max(int(actual.sum()+predicted.sum()), 1))
        return float(np.mean(scores))
    if metric in {"rocauc", "ap"}:
        p, y = p.reshape(len(y), -1), y.reshape(len(y), -1)
        if p.shape != y.shape:
            raise ValueError(f"Prediction/target shape mismatch: {p.shape}, {y.shape}")
        values = []
        for col in range(y.shape[1]):
            valid = np.isfinite(y[:, col]) & np.isfinite(p[:, col])
            yc, pc = y[valid, col], p[valid, col]
            if not (np.any(yc == 1) and np.any(yc == 0)):
                continue
            values.append((_auc if metric == "rocauc" else _ap)(yc, pc))
        return float(np.mean(values)) if values else float("nan")
    if metric in {"mae", "log10_mse"}:
        p, y = p.reshape(len(y), -1), y.reshape(len(y), -1)
        if p.shape != y.shape:
            raise ValueError(f"Prediction/target shape mismatch: {p.shape}, {y.shape}")
        valid = np.isfinite(y) & np.isfinite(p)
        if not valid.any():
            return float("nan")
        error = p[valid].astype(np.float64) - y[valid].astype(np.float64)
        return (float(np.mean(np.abs(error))) if metric == "mae" else
                float(np.log10(max(np.mean(error**2), np.finfo(np.float64).tiny))))
    if metric == "filtered_mrr":
        raise ValueError("Filtered MRR needs the graph and all node embeddings; use filtered_mrr")
    raise ValueError(f"Unsupported metric {metric!r}")


def link_ranks(graph, H, bb=None, exclude_self=False):
    """Per-positive reciprocal ranks with upstream LRGB filtering and tie order.

    Every other known positive destination for the same source is removed.
    Self candidates remain for mrr_filtered, matching the released evaluator.
    The optional no-self variant is explicitly separate.
    """
    if not torch.isfinite(H).all():
        raise FloatingPointError("Non-finite node embeddings in link evaluation")
    result = []
    batch = graph.batch_vec
    labels = graph.edge_label.reshape(-1)
    for g in range(graph.num_graphs):
        nodes = torch.where(batch == g)[0]
        if not len(nodes):
            continue
        mapping = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=H.device)
        mapping[nodes] = torch.arange(len(nodes), device=H.device)
        pos_mask = (labels == 1) & (batch[graph.edge_label_index[0]] == g)
        pos = mapping[graph.edge_label_index[:, pos_mask]]
        if not pos.shape[1]:
            continue
        scores = H[nodes] @ H[nodes].T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite link scores before filtering")
        positive_scores = scores[pos[0], pos[1]].clone()
        scores[pos[0], pos[1]] = -torch.inf
        if exclude_self:
            scores.fill_diagonal_(-torch.inf)
        candidates = torch.cat((positive_scores[:, None], scores[pos[0]]), dim=1)
        order = torch.argsort(candidates, dim=1, descending=True)
        ranks = torch.nonzero(order == 0, as_tuple=False)[:, 1] + 1
        result.append(ranks.to(H.dtype).reciprocal())
    return torch.cat(result) if result else H.new_empty(0)


def link_graph_metrics(graph, H, bb=None):
    """One filtered-MRR value per graph, including zero for no positive links."""
    if not torch.isfinite(H).all():
        raise FloatingPointError("Non-finite node embeddings in link evaluation")
    result = []
    batch = graph.batch_vec
    for g in range(graph.num_graphs):
        nodes = torch.where(batch == g)[0]
        e = graph.edge_label_index
        positive = (graph.edge_label.reshape(-1) == 1) & (batch[e[0]] == g)
        pos = e[:, positive]
        if not pos.shape[1]:
            result.append(H.new_tensor(0.))
            continue
        mapping = torch.full((graph.num_nodes,), -1, device=H.device, dtype=torch.long)
        mapping[nodes] = torch.arange(len(nodes), device=H.device)
        local = mapping[pos]
        scores = H[nodes] @ H[nodes].T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite link scores before filtering")
        pred_pos = scores[local[0], local[1]].clone()
        scores[local[0], local[1]] = -torch.inf
        order = torch.argsort(torch.cat((pred_pos[:, None], scores[local[0]]), 1), descending=True, dim=1)
        rank = torch.nonzero(order == 0, as_tuple=False)[:, 1] + 1
        result.append(rank.to(H.dtype).reciprocal().mean())
    return torch.stack(result)


def filtered_mrr(graph, H, bb=None):
    return float(link_graph_metrics(graph, H, bb).mean())


def metric_value(graph, H, bb, indices=None):
    if graph.task_type == "link_prediction":
        return filtered_mrr(graph, H, bb)
    pred, target = predictions(graph, H, bb), targets_for(graph)
    if indices is not None:
        pred, target = pred[indices], target[indices]
    return metric_from_predictions(graph, pred, target)
