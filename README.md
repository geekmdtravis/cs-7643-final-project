# ChestX-ray14 Multi-Modal Deep Learning Project

An advanced deep learning project for analyzing chest X-rays using both image data and clinical information through multi-modal approaches.

## Project Overview

This project implements various deep learning models for chest X-ray analysis, incorporating both image data and clinical information. It supports multiple state-of-the-art architectures including DenseNet and Vision Transformers (ViT), with both vanilla and multi-modal variants.

## Setup and Installation

### Prerequisites
- CUDA-capable GPU (Project uses CUDA 11.8) preferred, but not required.
- Conda package manager preferred, but any Python package manager should work.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/geekmdtravis/cs-7643-final-project.git
cd cs-7643-final-project
```

2. Create the conda environment:
```bash
conda env create -f environment.yaml
```

3. Activate the environment:
```bash
conda activate cs7643-project
```

## Project Structure

- `artifacts/` - Generated data files and model checkpoints
- `results/` - Training results, plots, and saved models
- `src/` - Source code for data processing, models, and utilities
- `tests/` - Unit tests

### Main Scripts

- `run_prepare_data.py` - Prepares dataset by splitting into train/val/test sets and creating clinical matrix-embedded images
- `run_trainer.py` - Handles model training with various architectures
- `run_inference.py` - Runs inference using trained models
- `run_calculate_dataset_stats.py` - Calculates dataset statistics for normalization

## Usage Guide

### 1. Data Preparation

Run the data preparation script:
```bash
python run_prepare_data.py
```
This will:
- Split data into train, validation, and test sets
- Create clinical matrix-embedded images
- Save processed data to the artifacts directory

### 2. Model Training

Train models using:
```bash
python run_trainer.py
```

Supported models:
- DenseNet Variants:
  - densenet121 / densenet121_mm
  - densenet201 / densenet201_mm
- Vision Transformer Variants:
  - vit_b_16 / vit_b_16_mm
  - vit_b_32 / vit_b_32_mm
  - vit_l_16 / vit_l_16_mm

Note: Models with '_mm' suffix are multi-modal variants that incorporate clinical data by augmenting the classification head.

Training parameters can be configured in the script:
- Learning rate: 1e-3 (default)
- Batch size: 32 (default)
- Epochs: 2 (default)
- Focal Loss: Enabled by default

### 3. Inference

Run inference using:
```bash
python run_inference.py
```

The script will:
- Load a trained model
- Run predictions on the test set
- Calculate and display evaluation metrics (AUC scores)

### 4. Dataset Statistics

Calculate dataset statistics using:
```bash
python run_calculate_dataset_stats.py
```

This generates mean and standard deviation values for dataset normalization.

## Project Links

- [Project Board](https://github.com/users/geekmdtravis/projects/4/views/1)
- [Repository](https://github.com/geekmdtravis/cs-7643-final-project)

## Dependencies

Key dependencies include:
- Python 3.12
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

For a complete list of dependencies, see `environment.yaml`.
