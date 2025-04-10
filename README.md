# ChestX-ray14 Multi-Modal Deep Learning Project
- [Project Link](https://github.com/users/geekmdtravis/projects/4/views/1)
- [Repo Link](https://github.com/geekmdtravis/cs-7643-final-project)
## 1. Overview

*   **Goal:** Improve Chest X-ray classification using the NIH ChestX-ray14 dataset by integrating patient clinical data (age, gender, view position, follow-up number) with deep learning image models.
*   **Models:** Compares baseline CNN (DenseNet-201) with multi-modal approaches. Includes implementations for DenseNet-201 and infrastructure (dependencies like `timm`) for adding Vision Transformers (ViT).
*   **Multi-Modal Strategies:** Explores two methods for integrating tabular data:
    *   **Feature Concatenation:** Combining image features extracted by a CNN/ViT with tabular features before a final classification layer (implemented in `DenseNet201MultiModal`).
    *   **Image Embedding:** Embedding normalized tabular data directly into a corner of the input image tensor before feeding it to the model (utility functions in `src/utils/image_manipulation.py`, particularly suited for ViTs).

## 2. Project Structure

*   `/src`: Core source code.
    *   `/data`: Dataset loading (`dataset.py`), data download (`download.py`), dataloader creation (`create_dataloaders.py`).
    *   `/models`: Model definitions (`densenet_201_vanilla.py`, `densenet_201_multimodal.py`). (ViT model to be added).
    *   `/utils`: Utility functions - configuration (`config.py`), data preprocessing (`preprocessing.py`), image manipulation (`image_manipulation.py` - padding & embedding), system info (`sytem_info.py`).
    *   `/notebooks`: Jupyter notebooks for experiments, analysis, or exploration (e.g., initial model training, visualization).
*   `/artifacts`: Stores processed data (e.g., `train.csv`, `test.csv`), embedded image examples.
*   `/logs`: Log files generated during runs (configured in `src/utils/config.py`).
*   `/results`: Stores model outputs.
    *   `/checkpoints`: Saved model weights.
    *   `/plots`: Performance plots (e.g., loss curves, ROC curves).
*   `/tests`: Unit tests for various components (e.g., `test_dataset.py`, `test_image_manipulation.py`).
*   `main.py`: Script for initial data download, preprocessing, and train/test split. (Future work: Extend to handle training/inference workflows).
*   `demo_*.py`: Example scripts demonstrating specific functionalities (e.g., `demo_pad_embed_image.py`, `demo_dataset_usage.py`).
*   `environment.yaml`: Conda environment file for Linux/Windows.
*   `environment_mac.yaml`: Conda environment file for macOS.
*   `.env`: Local configuration file (create from example below).
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `.flake8`: Configuration for the Flake8 linter.
*   `README.md`: This file.

## 3. Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with actual URL
    cd cs-7643-final-project
    ```
2.  **Create Conda Environment:**
    *   Use the appropriate environment file for your OS:
        ```bash
        # For Linux/Windows
        conda env create -f environment.yaml

        # For macOS
        conda env create -f environment_mac.yaml
        ```
    *   Activate the environment:
        ```bash
        conda activate cs7643-project
        ```
3.  **Configure Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Copy the following content into it and adjust values as needed for your system and experiments:
    ```bash
    # .env example content
    NUM_WORKERS=8       # Number of workers for data loading (adjust based on CPU cores)
    BATCH_SIZE=32       # Batch size for training/evaluation
    NUM_EPOCHS=10       # Number of epochs for training
    LEARNING_RATE=0.001 # Learning rate for optimizer
    OPTIMIZER=adam      # Optimizer to use (adam or sgd)
    DEVICE=cuda         # Device to use ('cuda' if GPU available, else 'cpu')
    LOG_LEVEL=info      # Logging level (debug, info, warning, error, critical)
    LOG_FILE=app.log    # Log file name (will be saved in /logs/)
    ```
4.  **Download and Prepare Data:**
    
    Not presently implemented in `main.py`, 
    look at the `demo_*` scripts.

## 4. Usage

    Not presently implemented in `main.py`, 
    look at the `demo_*` scripts.

*   **Data Setup:** Ensure you have run `python main.py` successfully (Step 3.4) to create `artifacts/train.csv` and `artifacts/test.csv`.
*   **Training & Inference (Current Status):**
    *   Currently, training and inference logic is not centralized in `main.py`.
    *   Look for specific training scripts or Jupyter notebooks within the `/src/notebooks/` directory to run experiments. These notebooks/scripts likely utilize the `ChestXrayDataset` from `src/data/dataset.py`, models from `src/models/`, and configuration from `src/utils/config.py`.
    *   *(Future Work: Implement command-line interface in `main.py` for streamlined training and inference, e.g., `python main.py --train --config config.yaml` or `python main.py --infer --checkpoint model.pth`)*
*   **Demo Scripts:**
    *   `demo_pad_embed_image.py`: Demonstrates padding images and embedding tabular data into the image tensor. Run with `python demo_pad_embed_image.py`. Generates example images in `/artifacts`.
    *   `demo_densenet_vanilla.py`: Shows basic inference using the vanilla DenseNet model on a sample ImageNet image (not ChestX-ray data). Run with `python demo_densenet_vanilla.py`.
    *   `demo_dataset_usage.py`: Illustrates how to instantiate and use the `ChestXrayDataset` and associated dataloaders. Run with `python demo_dataset_usage.py`.

## 5. Models & Multi-Modal Integration

*   **`DenseNet201Vanilla`:** (`src/models/densenet_201_vanilla.py`)
    *   Standard DenseNet-201 model using ImageNet pre-trained weights.
    *   Serves as an image-only baseline.
*   **`DenseNet201MultiModal`:** (`src/models/densenet_201_multimodal.py`)
    *   **Concatenation Approach:** Takes the DenseNet-201 image features, concatenates them with the input tabular data tensor, and passes them through a custom multi-layer classifier head.
*   **Vision Transformer (ViT) - Planned:**
    - Still selecting model, but likely a variant of DeiT.
    - Plan an embedding approach, and add a custom classifier head similar to the CNN. 

## 6. Future Work / .Contribution

*   Need to discuss. 