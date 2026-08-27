# PDG-GCN

PyTorch implementation of the **Process-Direction-Gated Graph Convolutional Network (PDG-GCN)** for geomorphic river-reach classification.

This repository contains only the proposed model. It does not include baseline architectures, dataset preprocessing, training workflows, evaluation metrics, plotting, or post hoc interpretability analyses.

## Architecture

PDG-GCN represents a river network as a directed graph. Every stored edge `(i, j)` points from an immediately upstream reach `i` to its immediately downstream reach `j`. The model:

1. projects reach attributes into a hidden representation;
2. aggregates upstream and downstream context separately;
3. conditions neighbor attention on hydrogeomorphic edge descriptors;
4. adaptively fuses the two directional representations;
5. retains local reach attributes through a residual pathway; and
6. applies batch normalization, ReLU, dropout, and a linear classifier.

For each reach, the feature-wise gate `g` is the **downstream allocation coefficient**. The directional fusion is

```text
g * downstream_context + (1 - g) * upstream_context + local_residual
```

The gate represents learned information allocation for prediction. It is not a direct measurement of physical or causal influence.

## Direction and edge conventions

`flow_edge_index` has shape `[2, num_edges]`. Each column stores a flow-oriented connection:

```text
immediately upstream reach -> immediately downstream reach
```

`edge_attr[e]` describes that same ordered pair. Under the manuscript formulation,

```text
e_ij = [x_i, x_j, x_j - x_i, abs(x_j - x_i), c_ij]
```

where `c_ij` contains structural indicators such as junction-related connectivity information. Signed differences must retain the upstream-to-downstream ordering even when the edge is used to obtain downstream contextual information.

Attention coefficients are normalized separately over the relevant upstream or downstream neighbor set of each receiving reach.

## Installation

```bash
git clone https://github.com/<OWNER>/PDG-GCN.git
cd PDG-GCN
pip install -e .
```

Python 3.9 or later and PyTorch 2.5 or later are required.

## Minimal use

```python
import torch
from pdg_gcn import PDGGCN

model = PDGGCN(
    node_channels=10,
    edge_channels=42,
    num_classes=6,
    hidden_channels=384,
    dropout=0.30,
)

node_features = torch.randn(100, 10)
flow_edge_index = torch.randint(0, 100, (2, 180), dtype=torch.long)
edge_attr = torch.randn(180, 42)

logits, direction_weights = model(
    node_features,
    flow_edge_index,
    edge_attr,
    return_direction_weights=True,
)

probabilities = logits.softmax(dim=-1)
# direction_weights[:, 0]: upstream allocation
# direction_weights[:, 1]: downstream allocation
```

The model returns logits so it can be used directly with classification losses that expect unnormalized scores.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests cover tensor shapes, directional allocation, receiver-wise attention normalization, upstream/downstream edge semantics, empty neighborhoods, input validation, and gradient flow.

## Citation

If you use this implementation, cite the accompanying manuscript:

> Wang, Z., Wang, J., and Pasternack, G. B. *Adaptive Representation of River-Network Context for National Geomorphic Stream Classification.*

Publication identifiers can be added to `CITATION.cff` when the manuscript DOI becomes available.

## License

Released under the MIT License.
