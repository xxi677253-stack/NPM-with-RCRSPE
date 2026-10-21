# NPM-Net: RCRSPE Code and Trained Weights

This repository accompanies the manuscript:

**A nonlinear-dynamics-prior-guided lightweight multitask neural network for flow pattern identification and water cut prediction in horizontal oil-water two-phase flow using a double-helical capacitance sensor**

It provides a standalone RCRSPE reference implementation, the NPM-Net architecture, and a trained PyTorch model state dictionary.

## Files

| File | Description |
| --- | --- |
| `rcrspe.py` | Reference implementation of refined composite rescaled range permutation entropy (RCRSPE). |
| `model.py` | NPM-Net architecture with MSB-Lite, DRB-Lite, RCRSPE-guided gating, and classification/regression heads. |
| `best_model.pth` | Trained NPM-Net weights saved as a PyTorch `state_dict`. |
| `README.md` | This file. |

## RCRSPE

The reference implementation follows the RCRSPE procedure described in Section 3.4, Eqs. (14)-(19), of the manuscript. Its default settings are **embedding dimension m = 6**, **time delay tau = 1**, and **20 scales**.

```python
import numpy as np
from rcrspe import rcrspe_vector

raw_voltage = np.sin(np.linspace(0, 16 * np.pi, 2000))  # Example signal only
features, coverage = rcrspe_vector(raw_voltage, m=6, tau=1, max_scale=20)
print(features.shape)  # (20,)
```

The implementation uses sample standard deviation (`ddof=1`), `epsilon=1e-12`, and stable sorting for ties. Because these numerical conventions were not fully specified in the manuscript, exact agreement with the previously computed training features has not been independently established. The original training workflow **loaded precomputed 20-dimensional RCRSPE vectors** rather than recalculating them during training.

## Trained NPM-Net weights

NPM-Net has **23,316 trainable parameters**. Its inputs are a normalized voltage window of shape `(batch, 1, 2000)` and an aligned 20-dimensional RCRSPE vector of shape `(batch, 20)`.

To load the provided weights:

```python
import torch
from model import FinalModel

model = FinalModel(num_classes=3, rspe_dim=20)
state_dict = torch.load("best_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

The reported software environment was Python 3.13 with PyTorch 2.9.0 in CPU mode. Computing RCRSPE also requires NumPy.

## Data availability and reproducibility limits

The original capacitance-sensor data and operating-condition labels are not included because they are proprietary research data.

The weight file contains model parameters only; it does **not** contain the fitted normalization/scaling parameters used in the training workflow. Therefore, the weight file alone is not sufficient to obtain valid real-data predictions or independently reproduce the paper's accuracy and regression metrics.

This release enables readers to inspect the RCRSPE reference algorithm, inspect the NPM-Net architecture, and load the trained weights. No dedicated CPU-latency benchmarking script or original test dataset is included; the paper's reported accuracy and latency figures are **not independently verified by this repository**.
