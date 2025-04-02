# 3D U-Net for Brain Tumor Segmentation (BraTS 2020)

This project implements a 3D U-Net model for brain tumor segmentation using the BraTS 2020 dataset. It includes data loading, preprocessing, model building, training, and evaluation.

## Project Overview

This repository contains:

* **Data Exploration and Loading:** Code to load, visualize, and analyze the BraTS 2020 dataset in `.h5` format.
* **Data Preprocessing:** Implementation of z-score normalization for MRI data.
* **3D U-Net Model:** Implementation of a 3D U-Net architecture using TensorFlow/Keras.
* **Data Generator:** A robust data generator for efficient patch extraction and data normalization.
* **Training and Validation:** Code for training the model with callbacks for checkpointing, learning rate reduction, and early stopping.

## Dataset

The model is trained on the BraTS 2020 dataset, which consists of multi-modal MRI scans and corresponding tumor segmentation masks. You will need to download the dataset and place it in the specified directory: `/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/`.

## Requirements

* Python 3.6+
* TensorFlow 2.x
* NumPy
* h5py
* Matplotlib
* nibabel

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install the required packages:

    ```bash
    pip install tensorflow numpy h5py matplotlib nibabel
    ```

3.  Download the BraTS 2020 dataset and place it in the `/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/` directory.

## Usage

1.  Open and run the Jupyter Notebook (`MRISegmentationProject.ipynb`).
2.  The notebook performs the following steps:
    * GPU detection and setup.
    * Data loading and visualization.
    * Data preprocessing.
    * Model building.
    * Data generation for training and validation.
    * Model training with callbacks.
    * Training completion message.

## Code Structure

* `MRISegmentationProject.ipynb`: Contains the main code for data loading, preprocessing, model building, and training.
* `data_generator()`: A function for generating 3D patches from the dataset.
* `build_3d_unet()` and `build_advanced_3d_unet()`: Functions for building the 3D U-Net model architecture.
* `preprocess_volume()`: A function for preprocessing the MRI volumes.
* The notebook also contains code for visualizing the data, finding tumor presence, and reconstructing 3D volumes.

## Model Details

* **Architecture:** 3D U-Net.
* **Input Shape:** (64, 64, 64, 4) (patch size)
* **Output Shape:** (64, 64, 64, 1) (segmentation mask)
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 50 (with early stopping)
* **Batch Size:** 2 (training), 1 (validation)

## Training Details

* The data is loaded and processed using the custom `data_generator`.
* The model is trained with callbacks for model checkpointing, learning rate reduction, and early stopping.
* Training and validation volumes are specified within the notebook.

## Future Improvements

* Implement Dice loss and Dice coefficient metrics for better segmentation performance.
* Add data augmentation to improve model robustness.
* Experiment with different model architectures and hyperparameters.
* Implement a separate test dataset for final model evaluation.
* Add visualization of segmentation results.
* Optimize memory usage.
* Add more detailed error handling.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

