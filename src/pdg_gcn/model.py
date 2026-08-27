"""Process-Direction-Gated Graph Convolutional Network (PDG-GCN)."""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["PDGGCN"]


def _grouped_softmax(scores: Tensor, groups: Tensor, num_groups: int) -> Tensor:
    """Apply a numerically stable softmax within integer-indexed groups."""
    if scores.numel() == 0:
        return scores

    group_max = scores.new_full((num_groups,), -torch.inf)
    group_max.scatter_reduce_(
        0,
        groups,
        scores,
        reduce="amax",
        include_self=True,
    )
    exp_scores = torch.exp(scores - group_max[groups])

    denominators = scores.new_zeros(num_groups)
    denominators.scatter_add_(0, groups, exp_scores)
    tiny = torch.finfo(scores.dtype).tiny
    return exp_scores / denominators[groups].clamp_min(tiny)


class _EdgeConditionedAggregation(nn.Module):
    """Aggregate one directional context from flow-oriented river edges.

    Every stored edge ``(i, j)`` is oriented from an immediately upstream reach
    ``i`` to its immediately downstream reach ``j``. For upstream context, ``i``
    sends a message to ``j``. For downstream context, ``j`` sends contextual
    information to ``i`` while the stored edge descriptor remains ``e_ij``.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        context: Literal["upstream", "downstream"],
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.context = context

        self.edge_score = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.message_projection = nn.Linear(
            hidden_channels,
            hidden_channels,
            bias=False,
        )

    def forward(
        self,
        hidden: Tensor,
        flow_edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        num_nodes = hidden.size(0)
        if flow_edge_index.size(1) == 0:
            return hidden.new_zeros((num_nodes, self.hidden_channels))

        flow_upstream, flow_downstream = flow_edge_index

        # Both scoring functions use the manuscript's flow-oriented ordered pair
        # (upstream reach, downstream reach, e_ij).
        score_inputs = torch.cat(
            (
                hidden[flow_upstream],
                hidden[flow_downstream],
                edge_attr,
            ),
            dim=-1,
        )
        scores = self.edge_score(score_inputs).squeeze(-1)

        if self.context == "upstream":
            neighbor = flow_upstream
            receiver = flow_downstream
        else:
            neighbor = flow_downstream
            receiver = flow_upstream

        attention = _grouped_softmax(scores, receiver, num_nodes).unsqueeze(-1)
        messages = attention * self.message_projection(hidden[neighbor])

        aggregated = hidden.new_zeros((num_nodes, self.hidden_channels))
        aggregated.index_add_(0, receiver, messages)
        return aggregated


class _DirectionGate(nn.Module):
    """Allocate complementary weights to upstream and downstream context."""

    def __init__(self, node_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.downstream_gate = nn.Sequential(
            nn.Linear(node_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, node_features: Tensor, hidden: Tensor) -> Tensor:
        downstream_weight = self.downstream_gate(
            torch.cat((hidden, node_features), dim=-1)
        )
        upstream_weight = 1.0 - downstream_weight
        return torch.stack((upstream_weight, downstream_weight), dim=-1)


class _PDGGCNLayer(nn.Module):
    """One manuscript-aligned PDG-GCN representation layer."""

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_projection = nn.Linear(node_channels, hidden_channels)

        self.upstream_aggregation = _EdgeConditionedAggregation(
            hidden_channels,
            edge_channels,
            context="upstream",
        )
        self.downstream_aggregation = _EdgeConditionedAggregation(
            hidden_channels,
            edge_channels,
            context="downstream",
        )
        self.direction_gate = _DirectionGate(node_channels, hidden_channels)

        self.local_residual = nn.Linear(node_channels, hidden_channels)
        self.hidden_normalization = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: Tensor,
        flow_edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Equation (1): h_i = W_x x_i + b_x.
        hidden = self.node_projection(node_features)

        upstream_context = self.upstream_aggregation(
            hidden,
            flow_edge_index,
            edge_attr,
        )
        downstream_context = self.downstream_aggregation(
            hidden,
            flow_edge_index,
            edge_attr,
        )

        # weights[..., 0] is upstream; weights[..., 1] is downstream.
        weights = self.direction_gate(node_features, hidden)
        fused = (
            weights[..., 1] * downstream_context
            + weights[..., 0] * upstream_context
            + self.local_residual(node_features)
        )

        representation = self.hidden_normalization(fused)
        representation = F.relu(representation)
        representation = self.dropout(representation)
        return representation, weights


class PDGGCN(nn.Module):
    """Process-Direction-Gated Graph Convolutional Network.

    Parameters
    ----------
    node_channels:
        Number of reach-scale variables in each node feature vector.
    edge_channels:
        Number of descriptors associated with each directed connection.
    num_classes:
        Number of output geomorphic stream-type classes.
    hidden_channels:
        Width of the hidden representation. The default is the configuration
        selected for the manuscript experiments.
    dropout:
        Dropout probability after batch normalization and ReLU. The default is
        the configuration selected for the manuscript experiments.

    Notes
    -----
    ``flow_edge_index[:, e] = (i, j)`` must represent a flow-oriented edge in
    which reach ``i`` is immediately upstream of reach ``j``.

    ``edge_attr[e]`` must describe the same ordered pair ``(i, j)``. Under the
    manuscript definition, ``e_ij = [x_i, x_j, x_j - x_i, |x_j - x_i|, c_ij]``.
    The model accepts any descriptor dimension but assumes that ordering and
    signs have already been constructed consistently.

    The learned gate is feature-wise. The downstream coefficient is ``g_i`` and
    the upstream coefficient is ``1 - g_i``. Returned reach-level directional
    weights are averages over hidden features and are ordered as
    ``(upstream, downstream)``.

    A reach without neighbors in one direction receives a zero contextual
    representation from that branch. Directional coefficients are not
    renormalized by neighborhood availability; the local residual remains.

    The forward method returns class logits. Apply Softmax outside the model
    only when probabilities are required.
    """

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        num_classes: int,
        hidden_channels: int = 384,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        for name, value in (
            ("node_channels", node_channels),
            ("edge_channels", edge_channels),
            ("num_classes", num_classes),
            ("hidden_channels", hidden_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.pdg_layer = _PDGGCNLayer(
            node_channels,
            edge_channels,
            hidden_channels,
            dropout,
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        node_features: Tensor,
        flow_edge_index: Tensor,
        edge_attr: Tensor,
        return_direction_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute geomorphic stream-type logits for all reaches.

        Parameters
        ----------
        node_features:
            Floating tensor of shape ``[num_nodes, node_channels]``.
        flow_edge_index:
            Integer tensor of shape ``[2, num_edges]``. Each edge is stored in
            immediately-upstream to immediately-downstream orientation.
        edge_attr:
            Floating tensor of shape ``[num_edges, edge_channels]`` describing
            the same ordered reach pairs as ``flow_edge_index``.
        return_direction_weights:
            If ``True``, also return reach-level allocation coefficients of
            shape ``[num_nodes, 2]`` in ``(upstream, downstream)`` order.
        """
        flow_edge_index, edge_attr = self._validate_and_align_inputs(
            node_features,
            flow_edge_index,
            edge_attr,
        )

        representation, featurewise_weights = self.pdg_layer(
            node_features,
            flow_edge_index,
            edge_attr,
        )
        logits = self.classifier(representation)

        if return_direction_weights:
            reach_weights = featurewise_weights.mean(dim=1)
            return logits, reach_weights
        return logits

    def _validate_and_align_inputs(
        self,
        node_features: Tensor,
        flow_edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if (
            node_features.ndim != 2
            or node_features.size(1) != self.node_channels
        ):
            raise ValueError(
                "node_features must have shape "
                f"[num_nodes, {self.node_channels}]."
            )
        if not node_features.is_floating_point():
            raise TypeError("node_features must use a floating-point dtype.")

        if flow_edge_index.ndim != 2 or flow_edge_index.size(0) != 2:
            raise ValueError("flow_edge_index must have shape [2, num_edges].")
        if flow_edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("flow_edge_index must use an integer dtype.")

        if (
            edge_attr.ndim != 2
            or edge_attr.size(0) != flow_edge_index.size(1)
            or edge_attr.size(1) != self.edge_channels
        ):
            raise ValueError(
                "edge_attr must have shape "
                f"[num_edges, {self.edge_channels}]."
            )
        if not edge_attr.is_floating_point():
            raise TypeError("edge_attr must use a floating-point dtype.")

        flow_edge_index = flow_edge_index.to(
            device=node_features.device,
            dtype=torch.long,
        )
        edge_attr = edge_attr.to(
            device=node_features.device,
            dtype=node_features.dtype,
        )

        if flow_edge_index.numel() > 0:
            if flow_edge_index.min() < 0 or flow_edge_index.max() >= node_features.size(0):
                raise IndexError("flow_edge_index contains an invalid node index.")

        return flow_edge_index, edge_attr
