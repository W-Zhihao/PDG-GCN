import torch

from pdg_gcn import PDGGCN
from pdg_gcn.model import _grouped_softmax


def _example_inputs():
    node_features = torch.randn(5, 4)
    flow_edge_index = torch.tensor(
        [[0, 1, 2, 2], [2, 2, 3, 4]],
        dtype=torch.long,
    )
    edge_attr = torch.randn(4, 6)
    return node_features, flow_edge_index, edge_attr


def test_output_and_direction_weight_shapes():
    x, edge_index, edge_attr = _example_inputs()
    model = PDGGCN(4, 6, 3, hidden_channels=8, dropout=0.0)
    model.eval()

    logits, weights = model(
        x,
        edge_index,
        edge_attr,
        return_direction_weights=True,
    )

    assert logits.shape == (5, 3)
    assert weights.shape == (5, 2)
    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5))


def test_empty_edge_set_is_finite():
    x = torch.randn(4, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 5))
    model = PDGGCN(3, 5, 2, hidden_channels=6, dropout=0.0)
    model.eval()

    logits, weights = model(
        x,
        edge_index,
        edge_attr,
        return_direction_weights=True,
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(weights).all()


def test_grouped_softmax_normalizes_each_receiver():
    scores = torch.tensor([1.0, 2.0, -1.0, 0.5])
    receivers = torch.tensor([2, 2, 3, 3])
    weights = _grouped_softmax(scores, receivers, num_groups=5)

    assert torch.allclose(weights[:2].sum(), torch.tensor(1.0))
    assert torch.allclose(weights[2:].sum(), torch.tensor(1.0))


def test_context_branches_follow_flow_orientation():
    model = PDGGCN(2, 1, 2, hidden_channels=2, dropout=0.0)
    upstream = model.pdg_layer.upstream_aggregation
    downstream = model.pdg_layer.downstream_aggregation

    for aggregation in (upstream, downstream):
        for parameter in aggregation.edge_score.parameters():
            torch.nn.init.zeros_(parameter)
        torch.nn.init.eye_(aggregation.message_projection.weight)

    hidden = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]
    )
    # 0 -> 2 and 1 -> 2 are upstream neighbors of 2; 2 -> 3 makes 3 the
    # downstream contextual neighbor of 2.
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]])
    edge_attr = torch.zeros(3, 1)

    upstream_context = upstream(hidden, edge_index, edge_attr)
    downstream_context = downstream(hidden, edge_index, edge_attr)

    assert torch.allclose(upstream_context[2], torch.tensor([2.0, 0.0]))
    assert torch.allclose(upstream_context[3], torch.tensor([0.0, 2.0]))
    assert torch.allclose(downstream_context[0], torch.tensor([0.0, 2.0]))
    assert torch.allclose(downstream_context[1], torch.tensor([0.0, 2.0]))
    assert torch.allclose(downstream_context[2], torch.tensor([0.0, 4.0]))


def test_gradients_reach_all_learnable_components():
    x, edge_index, edge_attr = _example_inputs()
    model = PDGGCN(4, 6, 3, hidden_channels=8, dropout=0.0)
    model.train()

    logits = model(x, edge_index, edge_attr)
    logits.square().mean().backward()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_input_shape_validation():
    model = PDGGCN(4, 6, 3, hidden_channels=8, dropout=0.0)
    x, edge_index, edge_attr = _example_inputs()

    try:
        model(x[:, :3], edge_index, edge_attr)
    except ValueError as error:
        assert "node_features" in str(error)
    else:
        raise AssertionError("Invalid node feature shape was not rejected.")
