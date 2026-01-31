# SpecGen Data Format

This document describes the data format required for training and inference with SpecGen.

## Directory Structure

```
data/
├── train.txt              # Training material names (one per line)
├── val.txt                # Validation material names (one per line)
├── merl_single.txt        # MERL RGB material names (optional)
├── rgb_train.txt          # RGB training material names (optional)
├── spectral/              # Spectral BRDF data
│   ├── material1_spec_ori.npy
│   ├── material2_spec_ori.npy
│   └── ...
├── merl/                  # MERL RGB BRDF data (optional)
│   ├── material1/
│   │   └── material1.npy
│   └── ...
├── renders/               # Rendered images for training
│   ├── material1.png
│   ├── material2.png
│   └── ...
└── renderdata/            # Geometry data for inference
    ├── normals.npy
    ├── mask.npy
    └── L.txt
```

---

## 1. BRDF Data Files

### 1.1 Spectral BRDF (`*_spec_ori.npy`)

**Format:** NumPy array with shape `(N, 5)`

Each row contains:
| Index | Field | Description | Range |
|-------|-------|-------------|-------|
| 0 | `theta_h` | Half-angle theta (Rusinkiewicz) | [0, 90) degrees |
| 1 | `theta_d` | Difference-angle theta | [0, 90) degrees |
| 2 | `phi_d` | Difference-angle phi | [0, 180) degrees |
| 3 | `wavelength_idx` | Wavelength index | [0, 38] (39 wavelengths) |
| 4 | `brdf_value` | BRDF reflectance value | [0, 1] normalized |

**Sample count:** ~5,120,000 samples per material

**Example:**
```python
import numpy as np

# Load spectral BRDF
data = np.load("material1_spec_ori.npy")
print(data.shape)  # (5120000, 5)

# Access one sample
theta_h, theta_d, phi_d, wavelength_idx, brdf_value = data[0]
```

### 1.2 MERL RGB BRDF (`*.npy`)

**Format:** NumPy array with shape `(N, 5)`

Same structure as spectral BRDF, but:
- `wavelength_idx` represents RGB channel index (0=R, 1=G, 2=B)
- BRDF values are mu-law encoded and normalized

**Sample count:** ~4,374,000 samples per material (90 × 90 × 180 × 3)

---

## 2. Training Images

### 2.1 Rendered Material Images (`*.png`)

**Format:** PNG image, RGB, 256×256 pixels

These are rendered images of a sphere with the target BRDF material, used as input to the Image Encoder.

**Preprocessing:**
```python
from PIL import Image
import numpy as np

# Load and normalize to [-1, 1]
image = Image.open("material.png")
image_array = np.array(image).astype(np.float32)
normalized = image_array / 127.5 - 1.0
```

---

## 3. Material List Files

### 3.1 `train.txt` / `val.txt`

Plain text files with one material name per line (without extension).

**Example `train.txt`:**
```
acrylic_felt_green
paper_blue
cc_iris_purple_gem
wood_oak
```

The actual BRDF file path is constructed as:
```python
brdf_path = f"{data_dir}/{material_name}_spec_ori.npy"
image_path = f"{render_dir}/{material_name}.png"
```

---

## 4. Inference Data (Geometry)

### 4.1 Normal Map (`normals.npy`)

**Format:** NumPy array with shape `(H, W, 3)`

Surface normal vectors in world coordinates, normalized to [-1, 1].

**Example:**
```python
normals = np.load("normals.npy")  # shape: (256, 256, 3)
# Each pixel: [nx, ny, nz] where ||n|| = 1
```

### 4.2 Mask (`mask.npy`)

**Format:** NumPy array with shape `(H, W)`, dtype `uint8`

Binary mask where:
- `1` = valid surface pixel
- `0` = background

### 4.3 Light Directions (`L.txt`)

**Format:** Text file with one light direction per line

Each line: `x y z` (normalized direction vector)

**Example `L.txt`:**
```
0.0 0.0 1.0
0.5773 0.5773 0.5773
-0.5 0.5 0.7071
```

---

## 5. Data Generation Scripts

### 5.1 Generate Normal Map from PNG

```python
# renderdata/getnpy.py
from PIL import Image
import numpy as np

def process_normal_map(input_path, normal_output_path, mask_output_path):
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img, dtype=np.uint8)
    
    # Map [0, 255] to [-1, 1]
    normal_map = (img_np.astype(np.float32) / 127.5) - 1.0
    np.save(normal_output_path, normal_map)
    
    # Generate mask (non-zero pixels)
    mask = np.any(img_np != 0, axis=-1).astype(np.uint8)
    np.save(mask_output_path, mask)
```

### 5.2 Convert MERL Binary to NPY

```python
# merlloader.py
import numpy as np

RED_SCALE = 1.0 / 1500.0
GREEN_SCALE = 1.15 / 1500.0
BLUE_SCALE = 1.66 / 1500.0

def mu_law_encode(x, mu=255):
    x = np.clip(x, -1, 1)
    return np.log1p(mu * np.abs(x)) / np.log1p(mu)

def read_merl_brdf(file_path):
    with open(file_path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        data = np.fromfile(f, dtype=np.float64)
    
    data = data.reshape((dims[0], dims[1], dims[2], 3))
    
    # Apply MERL scale factors
    data[..., 0] *= RED_SCALE
    data[..., 1] *= GREEN_SCALE
    data[..., 2] *= BLUE_SCALE
    
    # Normalize and apply mu-law encoding
    data = (data - data.min()) / (data.max() - data.min())
    for c in range(3):
        data[..., c] = mu_law_encode(data[..., c])
    
    return data
```

---

## 6. Rusinkiewicz Parameterization

The BRDF is parameterized using the Rusinkiewicz coordinate system:

```
         N (surface normal)
         |
         |  H (half vector)
         | /
         |/θh
    ─────●─────
        /|\
       / | \
      /  |  \
     L   |   V
    (light) (view)
```

**Angles:**
- `θh` (theta_h): Angle between half-vector H and surface normal N
- `θd` (theta_d): Angle between difference vector D and H
- `φd` (phi_d): Azimuthal angle of D around H

**Conversion:**
```python
from coords import get_rusinkiewicz_angles

# wi: incident light direction
# wo: outgoing view direction
# n: surface normal (usually [0, 0, 1] in local frame)
theta_h, theta_d, phi_d = get_rusinkiewicz_angles(wi, wo, n)
```

---

## 7. Quick Start: Prepare Your Own Data

### Step 1: Prepare BRDF Measurements

Convert your BRDF measurements to the required format:
```python
import numpy as np

# Your BRDF data should be sampled at these resolutions:
# theta_h: 90 bins (0-89 degrees)
# theta_d: 90 bins (0-89 degrees)  
# phi_d: 180 bins (0-179 degrees)
# wavelength: 39 bins (for spectral) or 3 bins (for RGB)

samples = []
for th in range(90):
    for td in range(90):
        for pd in range(180):
            for wl in range(39):  # or range(3) for RGB
                value = your_brdf_function(th, td, pd, wl)
                samples.append([th, td, pd, wl, value])

data = np.array(samples, dtype=np.float32)
np.save("your_material_spec_ori.npy", data)
```

### Step 2: Render Training Image

Render a 256×256 image of a sphere with your BRDF material under fixed lighting.

### Step 3: Create Material List

```bash
echo "your_material" >> train.txt
```

### Step 4: Train

```python
from core import train_main

train_main(
    data_dir="./data/spectral",
    image_dir="./data/renders",
    train_file="./data/train.txt",
    val_file="./data/val.txt",
    checkpoint_dir="./checkpoints"
)
```

---

## 8. Data Statistics

| Dataset | Materials | Samples/Material | Total Samples |
|---------|-----------|------------------|---------------|
| Spectral BRDF | ~50 | 5,120,000 | ~256M |
| MERL RGB | ~100 | 4,374,000 | ~437M |

**Grid Resolution:**
- θh: 90 bins
- θd: 90 bins
- φd: 180 bins
- Wavelength: 39 bins (spectral) / 3 bins (RGB)

---

## References

- [MERL BRDF Database](https://www.merl.com/brdf/)
- [Rusinkiewicz Parameterization](https://graphics.stanford.edu/papers/brdf/brdf.pdf)
- SpecGen: WACV 2026
