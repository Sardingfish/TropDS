# TropDS

![](https://img.shields.io/badge/version-1.0.0-green.svg) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-BSD3Clause-yellow.svg)](https://opensource.org/licenses/MIT)

TropDS: Downscaling to Enhance Tropospheric Delay Grid Precision in Space Geodesy

This project implements a deep learning-based grid-wise tropospheric delay data downscaling framework for high-precision space geodetic data processing. The model uses a U-Net architecture with physical constraints to restore blurred (low-resolution) tropospheric delay images to their clear (high-resolution) versions.

### Key Features

- **U-Net Architecture**: Encoder-decoder structure with skip connections for accurate image restoration
- **Physical Constraints**: Softplus activation ensures non-negative output (physically meaningful for tropospheric delay)
- **Spatial Weighting**: Historical RMSE-based weighting focuses training on high-error regions
- **Freeze Mechanism**: Low-RMSE regions (e.g., high-latitude/high-altitude) are frozen to prevent over-correction
- **Mixed Precision Support**: Optimized for efficient GPU utilization

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory (16GB recommended for optimal performance)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Sardingfish/TropDS.git
cd TropDS
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## How to Run

### Data Preparation

1. Place your grid-wise tropospheric delay data files in the `./data/` directory
2. Expected file format: NumPy `.npy` files
3. Expected data shape: `(180, 360)` for single sample or `(N, 180, 360)` for batches
4. File naming convention:
   - `{year}_blur.npy`: Blurred tropospheric delay data
   - `{year}_clear.npy`: Clear/reference tropospheric delay data

Example data structure:
```
data/
├── 2020_blur.npy
├── 2020_clear.npy
├── 2021_blur.npy
├── 2021_clear.npy
├── ...
├── 2024_blur.npy
└── weight.npy          # Optional: RMSE weight map (180x360)
```

### Training

1. **Configure training parameters** in `train.py`:
```python
# Data configuration
YEARS_TRAIN = [2020, 2021, 2022, 2023]  # Training years
YEAR_VAL = 2024                         # Validation year

# Training configuration
BATCH_SIZE = 32                         # Adjust based on GPU memory
LEARNING_RATE = 1e-4
EPOCHS = 200
PATIENCE = 20

# Freeze configuration
FREEZE_THRESHOLD = 0.01                 # Freeze RMSE < 0.01 regions
ENABLE_FREEZE = True
```

2. **Run training**:
```bash
python train.py
```

3. **Outputs** are saved to `./output/`:
   - `best_model.pth`: Best model checkpoint
   - `final_model.pth`: Final model checkpoint
   - `training_history.npy`: Training metrics history
   - `training_report.txt`: Training summary report

### Inference

1. **Configure inference parameters** in `inference.py`:
```python
MODEL_PATH = './output/best_model.pth'
YEAR_INFERENCE = 2025
```

2. **Run inference**:
```bash
python inference.py
```

3. **Output**:
   - `./output/{year}_deblurred.npy`: Deblurred tropospheric delay data

## File Structure

```
TropDS/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── train.py                  # Training script
├── inference.py              # Inference script
├── models.py                 # U-Net model definition
├── datasets.py               # Data loading and preprocessing
├── losses.py                 # Custom loss functions
├── metrics.py                # Evaluation metrics
├── data/                     # Input data directory
│   ├── {year}_blur.npy
│   ├── {year}_clear.npy
│   └── weight.npy            # Optional RMSE weight map
└── output/                   # Output directory
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.npy
    ├── training_report.txt
    └── {year}_deblurred.npy
```

## Model Architecture

```
U-Net with Physical Constraints
==============================
Input:  (B, 1, 180, 360)
Output: (B, 1, 180, 360)

Encoder Path:
  - enc1: ConvBlock(1, 32)
  - enc2: EncoderBlock(32, 64)    - MaxPool
  - enc3: EncoderBlock(64, 128)   - MaxPool
  - enc4: EncoderBlock(128, 256)  - MaxPool
  - enc5: EncoderBlock(256, 512)  - MaxPool
  - bottleneck: ConvBlock(512, 512)

Decoder Path:
  - dec4: DecoderBlock(512, 256, 256)  - UpConv + Skip
  - dec3: DecoderBlock(256, 128, 128)  - UpConv + Skip
  - dec2: DecoderBlock(128, 64, 64)    - UpConv + Skip
  - dec1: DecoderBlock(64, 32, 32)     - UpConv + Skip

Output Layer:
  - Conv2d(32, 1, kernel_size=1)
  - Softplus()  # Physical constraint: output > 0

Total Parameters: ~31M
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ding2026TropDS,
  title = {TropDS: Downscaling to Enhance Tropospheric Delay Grid Precision in Space Geodesy},
  author = {Junsheng Ding et al.},
  journal={Preprint},
  year={2026},
  publisher={****}
}
```

## License

This project is licensed under the BSD 3-Clause License.

## Acknowledgments

- U-Net architecture adapted from [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- SSIM implementation inspired by [pytorch-ssim](https://github.com/pytorch/pytorch/issues/2151)
