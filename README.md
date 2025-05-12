# Computer Vision Final Assessment: Heroes of Might and Magic 3 Unit Recognition

This repository is my final assessment for the Computer Vision course. The project focuses on creating a balanced dataset of units from the game "Heroes of Might and Magic 3" and training a neural network to recognize these units in images and videos.

---

## Project Goals

- **Dataset Creation:** Extract and annotate unit sprites from "Heroes of Might and Magic 3".
- **Balanced Dataset:** Ensure fair representation of all unit classes.
- **Neural Network Training:** Train a model for robust unit recognition.
- **Evaluation:** Test the model on screenshots and gameplay videos.

---

## Input Data

- **Sprites:** ["Heroes of Might and Magic 3" Sprites](https://www.spriters-resource.com/pc_computer/heroes3/)
- **Annotations:** ["Heroes of Might and Magic 3 Units" Kaggle Dataset](https://www.kaggle.com/datasets/daynearthur/heroes-of-might-and-magic-3-units/data)

---

## Workflow Overview

### Task 1: Dataset Preparation

- **Script:** `task-1_prepare_dataset.py`
- **Description:** Prepares raw sprite and annotation data for further processing. Handles file organization, renaming, and initial cleaning.
- **Output:** Cleaned and organized sprite and annotation files.

### Task 2: Generate the Dataset

- **Script:** `task-2_generate_dataset.py`
- **Description:** Splits the data into training, validation, and test sets, ensuring balanced class distribution across all splits. Copies images and generates CSV annotation files for each split.
- **Output:** `/dataset/train`, `/dataset/val`, `/dataset/test` folders and corresponding CSVs.

### Task 3: Inspect Dataset

- **Script:** `task-3_dataset_inspection.py`
- **Description:** Provides visualizations and statistics about the dataset, such as class distribution, frame sizes, and random sample images. Helps verify dataset balance and quality.
- **Output:** Visualizations saved in `/visualization_output`.

### Task 4: Train Neural Network

- **Script:** `task-4_train_network.py`
- **Description:** Defines and trains a convolutional neural network using TensorFlow/Keras. Uses the generated dataset for training and validation. Saves the trained model for later inference.
- **Output:** Trained model file (`unit_classifier_model.keras`), training history plots.

---

## Project Progress Flow

A detailed step-by-step flow of my progress and decisions is documented in [`progress_flow.md`](progress_flow.md).

## Dataset Overview

For a detailed overview of the dataset, including its structure, class distribution, and sample images, refer to the [`dataset_overview.md`](dataset_overview.md) file.

---

## Acknowledgements

- Sprites: [The Spriters Resource](https://www.spriters-resource.com/pc_computer/heroes3/)
- Annotations: [Kaggle Dataset by Dayne Arthur](https://www.kaggle.com/datasets/daynearthur/heroes-of-might-and-magic-3-units/data)

---

## Note on AI Assistance

This project was developed with help from GitHub Copilot, an AI coding assistant. Due to my limited experience with Python, Copilot played a crucial role in helping me write efficient code, troubleshoot issues, and complete the project on time.

---