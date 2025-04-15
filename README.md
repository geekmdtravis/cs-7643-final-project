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
    *   `/utils`: Utility functions - configuration (`config.py`), data preprocessing (`preprocessing.py`), image manipulation (`image_manipulation.py` - embedding), system info (`sytem_info.py`), path utilities (`path_utils.py`).
    *   `/notebooks`: Jupyter notebooks for experiments, analysis, or exploration (e.g., initial model training, visualization).
*   `/artifacts`: Stores processed data (`train.csv`, `test.csv`), original images (`cxr_train/`, `cxr_test/`), embedded images (`embedded_train/`, `embedded_test/`), and demo outputs (`demo/`).
*   `/logs`: Log files generated during runs (configured in `src/utils/config.py`).
*   `/results`: Stores model outputs.
    *   `/checkpoints`: Saved model weights.
    *   `/plots`: Performance plots (e.g., loss curves, ROC curves).
*   `/tests`: Unit tests for various components (e.g., `test_dataset.py`, `test_image_manipulation.py`).
*   `main.py`: Script for initial data download and a *basic* train/test split (primarily for quick setup). Use `prepare_data.py` for full processing.
*   `prepare_data.py`: **Main script for data preparation.** Downloads, performs robust train/test split with imputation/normalization, copies original images, and creates embedded images required for training.
*   `calculate_dataset_stats.py`: Utility script to compute mean and standard deviation for dataset-specific normalization.
*   `demo_*.py`: Example scripts demonstrating specific functionalities (see Usage section).
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
    *   Run the main data preparation script:
        ```bash
        python prepare_data.py
        ```
    *   This script performs the following essential steps:
        1.  Downloads the NIH ChestX-ray14 dataset using the Kaggle API (if not already downloaded to the cache).
        2.  Loads the clinical data (`Data_Entry_2017_v2020.csv`).
        3.  Performs preprocessing, imputation, and normalization on the clinical data.
        4.  Splits the data into training and testing sets, saving `artifacts/train.csv` and `artifacts/test.csv`.
        5.  Copies the corresponding original PNG images from the Kaggle cache to `artifacts/cxr_train/` and `artifacts/cxr_test/`.
        6.  Creates versions of the images with clinical data embedded in the top-left corner, saving them to `artifacts/embedded_train/` and `artifacts/embedded_test/`. These are used for the image embedding multi-modal approach.
    *   *(Note: `python main.py` can be run for a quicker initial download and basic split, but `prepare_data.py` creates the fully processed data required by the dataloaders and embedding methods).*

## 4. Usage

*   **Data Setup:** Ensure you have successfully run `python prepare_data.py` (Step 3.4) to generate the necessary CSV files and image directories within `/artifacts`.
*   **Training & Inference (Current Status):**
    *   Currently, training and inference logic is not centralized.
    *   Look for specific training scripts or Jupyter notebooks within the `/src/notebooks/` directory to run experiments.
    *   These notebooks/scripts will typically use:
        *   `src.data.create_dataloader` to load data (using images from `artifacts/embedded_train` or `artifacts/cxr_train`).
        *   Models defined in `src/models/`.
        *   Configuration loaded via `src/utils/config.py` (reading from `.env`).
    *   *(Future Work: Implement a command-line interface, potentially in `main.py`, for streamlined training and inference, e.g., `python main.py --mode train --config config.yaml` or `python main.py --mode infer --checkpoint model.pth`)*
*   **Demo Scripts:**
    *   `demo_dataloader.py`: Demonstrates how to use `create_dataloader` to load the processed data (specifically from `artifacts/embedded_train`). It processes batches with different normalization settings ('none', 'dataset_specific', 'imagenet') and saves sample output images to `artifacts/demo/` for visualization. Run with `python demo_dataloader.py`.
    *   `demo_densenet_vanilla.py`: Shows basic inference using the `DenseNet201Vanilla` model on a generic sample ImageNet image (not ChestX-ray data) to verify the model class works. Run with `python demo_densenet_vanilla.py`.
    *   `calculate_dataset_stats.py`: Computes the mean and standard deviation of the training dataset (grayscale pixel values from `artifacts/embedded_train`). Useful for the `dataset_specific` normalization mode. Run with `python calculate_dataset_stats.py`.

## 5. Models & Multi-Modal Integration

*   **`DenseNet201Vanilla`:** (`src/models/densenet_201_vanilla.py`)
    *   Standard DenseNet-201 model using ImageNet pre-trained weights.
    *   Serves as an image-only baseline.
*   **`DenseNet201MultiModal`:** (`src/models/densenet_201_multimodal.py`)
    *   **Concatenation Approach:** Takes the DenseNet-201 image features, concatenates them with the input tabular data tensor, and passes them through a custom multi-layer classifier head.
*   **Vision Transformer (ViT) - Planned:**
    - Still selecting model, but likely a variant of DeiT.
    *   Plan an embedding approach (using `src.utils.image_manipulation.embed_clinical_data_into_image` during data preparation via `prepare_data.py`) and add a custom classifier head similar to the CNN.

## 6. Future Work / Contribution

*   Implement Vision Transformer (ViT) models (e.g., DeiT) and integrate them using the image embedding approach.
*   Develop centralized training and inference scripts/workflows (e.g., extending `main.py` or creating new scripts like `train.py`, `infer.py`).
*   Conduct and document comprehensive experiments comparing baseline, concatenation, and embedding approaches across different model backbones (CNN, ViT).
*   Expand evaluation metrics (e.g., precision, recall, F1-score per class, ROC AUC).
*   Refine data preprocessing and augmentation strategies.
*   Add more comprehensive unit and integration tests.
*   Explore other multi-modal fusion techniques.
