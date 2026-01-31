# SpecGen: Single-Image Spectral BRDF Prediction via K-Planes HyperNetwork

Official implementation of **SpecGen** (WACV 2026).

## Overview

SpecGen predicts spectral BRDF (Bidirectional Reflectance Distribution Function) from a single input image using a HyperNetwork architecture based on K-Planes representation.

### Pipeline

```
Input Image → CNN Encoder → MLP generates 6 K-Planes → Bilinear Interpolation → SKNet Feature Fusion → Decoder → Spectral BRDF
```

## Project Structure

```
core/
├── __init__.py      # Module initialization
├── model.py         # KPlaneField model definition (core)
├── train.py         # Training script
├── inference.py     # Inference/rendering script
├── ops.py           # Activation functions and interpolation
├── coords.py        # Rusinkiewicz coordinate transformation
├── config.py        # Configuration file
├── README.md        # This file
├── data/
│   └── README.md    # Detailed data format documentation
└── renderdata/      # Sample geometry data for inference
    ├── normals.npy
    ├── mask.npy
    └── L.txt
```

## Model Architecture

### 1. Image Encoder

A CNN that encodes the input image into a compact feature representation:

```python
Conv2d(3→16) → ReLU → Conv2d(16→32) → ReLU → Conv2d(32→20) → ReLU
# Output: (batch, 20, 64, 64)
```

### 2. K-Planes Generation (6 MLPs)

The encoded features are passed through 6 separate MLPs to generate 6 feature planes:

| MLP | Output Shape | Corresponding Dimensions |
|-----|-------------|-------------------------|
| mlp0 | (64, 90, 90) | θ_h × θ_d |
| mlp1 | (64, 180, 90) | φ_d × θ_h |
| mlp2 | (64, 39, 90) | λ × θ_h |
| mlp3 | (64, 180, 90) | φ_d × θ_d |
| mlp4 | (64, 39, 90) | λ × θ_d |
| mlp5 | (64, 39, 180) | λ × φ_d |

where λ represents the wavelength dimension for spectral BRDF.

### 3. Feature Fusion (SKNet)

Attention-weighted fusion of interpolated features from all 6 planes using SKNet-style selective kernel mechanism.

### 4. Decoder

```python
sigma_net: 64 → 32 dim features
color_net: 32 → 1 (BRDF value)
```

## Rusinkiewicz Parameterization

We use the Rusinkiewicz parameterization for BRDF, which describes the lighting geometry with 3 angles:

- **θ_h**: Half-vector elevation angle [0°, 90°]
- **θ_d**: Difference-vector elevation angle [0°, 90°]
- **φ_d**: Difference-vector azimuthal angle [0°, 180°]

## Getting Started

### Requirements

```bash
pip install torch numpy pillow scipy tqdm tensorboard
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Training

```python
from core import train_main

train_main(
    config_path="core/config.py",
    train_file="train.txt",
    val_file="val.txt",
    image_dir="./images",
    data_dir="./data",
    checkpoint_dir="./checkpoints",
    batch_size=2048,
    num_epochs=15
)
```

### Inference

```python
from core import load_model, render_single_image
import torch

device = torch.device("cuda")
model = load_model("checkpoints/best.pth", "core/config.py", device)

render_single_image(
    model=model,
    img_path="input.png",
    N_map_file="normals.npy",
    mask_file="mask.npy",
    L_file="light.txt",
    out_dir="./output",
    obj_name="test",
    device=device
)
```

## Data Format

For detailed data format documentation, see **[data/README.md](data/README.md)**.

### Quick Summary

| Data Type | Format | Shape/Content |
|-----------|--------|---------------|
| Input Image | PNG | 256×256 RGB sphere render |
| BRDF Data | `.npy` | `(N, 5)` - [θh, θd, φd, λ, value] |
| Normal Map | `.npy` | `(H, W, 3)` normalized normals |
| Mask | `.npy` | `(H, W)` binary mask |
| Light Dirs | `.txt` | one `x y z` direction per line |

## Loss Function

`SpectralLoss` consists of three components:

1. **MSE Loss**: Mean squared error between prediction and ground truth
2. **Scale Loss**: Scale-invariant constraint
3. **TV Loss**: Total variation regularization for smoothness

```python
loss = mse_weight * mse_loss + scale_weight * scale_loss + tv_weight * tv_loss
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{specgen2026,
    title={SpecGen: Neural Spectral BRDF Generation via Spectral-Spatial Tri-plane Aggregation},
    author={},
    booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year={2026}
}
```

## Acknowledgements

This project builds upon the K-Planes framework:

```bibtex
@inproceedings{kplanes_2023,
    title={K-Planes: Explicit Radiance Fields in Space, Time, and Appearance},
    author={Sara Fridovich-Keil and Giacomo Meanti and Frederik Rahbæk Warburg 
            and Benjamin Recht and Angjoo Kanazawa},
    booktitle={CVPR},
    year={2023}
}
```

## License

This project is released under the MIT License.

## Notes

1. The model requires CUDA support via `tiny-cuda-nn`
2. Mixed precision training (`autocast`) is recommended
3. `avg_flag` switches between Spectral/RGB modes
