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
- **Description:** Splits the data into training, validation, and test sets, ensuring balanced class distribution and unique `Frame_ID` across all splits. Copies images and generates CSV annotation files for each split.
- **Output:** `/dataset/train`, `/dataset/val`, `/dataset/test` folders and corresponding CSVs.

### Task 3: Inspect Dataset

- **Script:** `task-3_dataset_inspection.py`
- **Description:** Provides visualizations and statistics about the dataset, such as class distribution, frame sizes, and random sample images. Helps verify dataset balance and quality.
- **Output:** Visualizations saved in `/visualization_output` (see screenshots for examples).

### Task 4: Train Neural Network

- **Script:** `task-4_train_network.py`
- **Description:** Defines and trains a convolutional neural network using TensorFlow/Keras. Uses the generated dataset for training and validation. Saves the trained model for later inference.
- **Output:** Trained model file (`unit_classifier_model.keras`), training history plots.

### Task 5: Test and Inference

- **Script:** `task-5_inference.py`
- **Description:** Loads the trained model and performs inference on test images, screenshots, or gameplay videos. Visualizes predictions by drawing bounding boxes and class labels.
- **Output:** Annotated images/videos, accuracy metrics, and per-class evaluation.

### Task 6: Advanced Unit Detection

- **Scripts:**
  - `task-6_find_units_image.py`: Detects and highlights units in a single image using the trained model.
  - `task-6_find_units_video.py`: Detects and tracks units in a video, combining periodic recognition with object tracking for efficiency.
  - `task-6_find_units_template_matching.py`: Alternative approach using template matching for unit detection (no neural network required).
- **Description:** These scripts implement robust and efficient unit detection for both images and videos, with optimizations for speed and accuracy.
- **Output:** Annotated images/videos with detected units.

---

## How to Run

1. **Prepare the dataset:**  
   Run `task-1_prepare_dataset.py` and `task-2_generate_dataset.py` to organize and split the data.

2. **Inspect the dataset:**  
   Run `task-3_dataset_inspection.py` to visualize and verify the dataset.

3. **Train the neural network:**  
   Run `task-4_train_network.py` to train the model.

4. **Test and inference:**  
   Use `task-5_inference.py` for evaluation and predictions.

5. **Advanced detection:**  
   Use the scripts in Task 6 for robust detection in images and videos.

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