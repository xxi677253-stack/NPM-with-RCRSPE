# NPM-Net: RCRSPE Implementation and Trained Weights

This repository provides the RCRSPE implementation and trained NPM-Net weights associated with the manuscript:

**A nonlinear-dynamics-prior-guided lightweight multitask neural network for flow pattern identification and water cut prediction in horizontal oil-water two-phase flow using a double-helical capacitance sensor**

## Repository Contents

| File             | Description                                                                                                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rcrspe.py`      | Python reference implementation of refined composite rescaled range permutation entropy (RCRSPE), used to extract 20-dimensional nonlinear-dynamics features from voltage signals. |
| `model.py`       | PyTorch implementation of NPM-Net, including MSB-Lite, DRB-Lite, RCRSPE-guided feature modulation, and multitask prediction heads.                                                 |
| `best_model.pth` | Trained model weights for NPM-Net, which contains 23,316 trainable parameters.                                                                                                     |
| `README.md`      | Overview of the repository contents.                                                                                                                                               |

## Implementation Details

The RCRSPE implementation follows the method described in Section 3.4 of the manuscript, with an embedding dimension of **m = 6**, a time delay of **τ = 1**, and **20 scales**.

NPM-Net takes a 2,000-point voltage window and a corresponding 20-dimensional RCRSPE feature vector as inputs. It jointly performs flow pattern classification and water cut prediction.

The code is implemented in Python using NumPy and PyTorch.
