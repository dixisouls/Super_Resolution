# Image Super Resolution with EDSR and Channel Attention

An advanced image super resolution system implementing the **Enhanced Deep Super Resolution (EDSR)** architecture with **Channel Attention** mechanism. Built with PyTorch, this project provides multiple model variants and supports up to 8x upscaling of images while maintaining high quality and detail.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Models](#models)
- [Docker Repository](#docker-repository)
- [Results](#results)

---

## Overview

The goal of this project is to enhance image resolution using deep learning techniques. The system implements three variants of the EDSR model:
1. **Basic EDSR**: A lightweight implementation for quick results
2. **Deep EDSR**: An enhanced version with more residual blocks
3. **EDSR with Channel Attention**: The most advanced variant incorporating channel attention mechanism for better feature extraction

### Key Features:
1. **High Upscaling Factor**: Supports up to 8x image upscaling
2. **Channel Attention**: Enhanced feature extraction using attention mechanism
3. **Multiple Model Variants**: Choice of models based on requirements
4. **Configurable Architecture**: Easy to modify model parameters

---

## Features

- **Multiple EDSR Variants** for different use cases
- **Channel Attention Mechanism** for improved feature learning
- **Configurable Training Parameters** including batch size, learning rate, and epochs
- **Efficient Data Loading** with custom dataset implementation
- **Comprehensive Logging** for both training and inference
- **Easy-to-use Inference Mode** for single image super resolution

---

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- Pillow
- tqdm

Clone the repository:

```bash
git clone https://github.com/dixisouls/Super_Resolution.git
```

Install the required packages:

```bash
pip install torch torchvision tqdm Pillow
```

---

## Usage

### Training

To train the super resolution model:

```bash
python train.py
```

The training script will:
- Load configuration from `config.py`
- Set up logging in the `logs` directory
- Save model checkpoints in `trained_models` directory
- Display progress with a progress bar

### Inference

To upscale an image:

```bash
python inference.py path/to/your/image.jpg
```

The inference script will:
- Load the best model from `trained_models/best.pth`
- Process the input image
- Save the super-resolved output as `output.png`

---

## Project Structure

```
project/
├── models/
│   ├── edsr.py                 # Basic EDSR implementation
│   ├── edsr_deep.py           # Deep EDSR variant
│   └── edsr_channel_attention.py  # EDSR with attention
├── utils/
│   ├── data_utils.py          # Dataset and dataloader utilities
│   ├── infer_utils.py         # Inference helper functions
│   └── utils.py               # General utilities
├── trained_models/
│   └── best.pth               # Trained model weights
├── config.py                  # Training configuration
├── infer_config.py           # Inference configuration
├── train.py                  # Training script
└── inference.py              # Inference script
```

---

## Configuration

### Training Configuration (`config.py`)

- **Data Settings**
  - `DATA_DIR`: Base data directory
  - `HIGH_RES_DIR`: High resolution images directory
  - `LOW_RES_DIR`: Low resolution images directory

- **Model Parameters**
  - `SCALE_FACTOR`: Upscaling factor (default: 8)
  - `NUM_FEATURES`: Number of feature channels
  - `NUM_RES_BLOCKS`: Number of residual blocks
  - `REDUCTION_RATIO`: Channel attention reduction ratio

- **Training Parameters**
  - `BATCH_SIZE`: Training batch size
  - `EPOCHS`: Number of training epochs
  - `LEARNING_RATE`: Model learning rate

### Inference Configuration (`infer_config.py`)

- **Model Settings**
  - `MODEL_PATH`: Path to trained model
  - `SCALE_FACTOR`: Upscaling factor
  - `OUTPUT_IMAGE_PATH`: Path for saving output

---

## Models

### 1. Basic EDSR
- Lightweight implementation
- 4 convolutional layers
- Suitable for quick testing

### 2. Deep EDSR
- Enhanced architecture
- Multiple residual blocks
- Better quality for general use

### 3. EDSR with Channel Attention
- Advanced architecture
- Channel attention mechanism
- Best quality for detailed images

---

## Docker Repository

The Docker repository for this project can be found here:

[**Docker Repository**](https://hub.docker.com/repository/docker/dixisouls/super_resolution/general)

---

## Results

The model achieves significant quality improvements in image super-resolution:
- Supports up to 8x upscaling
- Maintains sharp edges and textures
- Reduces artifacts and blur

For best results, use the EDSR with Channel Attention model.

---
