
# Emotion Recognition Project

This repository contains two main components:

1. **Model Training and Evaluation** (`1_model_code/`)


---
### Contents:
- `train.py`: Script for training the model.
- `test.py`: Script for evaluating the model on the test set.

### Configuration:
- Paths to the dataset and pretrained weights are already set within the respective Python files.

### Setup:
Install required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Usage:

To **train** the model:
```bash
python train.py
```

To **evaluate** the model on the test set:
```bash
python test.py
```
**Model weight** ```model_best.pt``` download from the given link below
- **link** : https://drive.google.com/file/d/1QjYidu2SFmCRjFiTvYGOfiSM2lOkfFGq/view?usp=sharing


# SpEmoC Dataset Submission

This repository contains the dataset and resources for the **SpEmoC** project. Follow the instructions below to extract and organize the dataset files properly.

---

## 📦 Dataset Archives

Inside the `SpEmoC_dataset_submission` folder, you will find the following archive files:

- `test.zip` – Contains:
  - `test_set_videos/` – Test set videos
  - `test_json/` – Annotations for the test set

- `val.zip` – Contains:
  - `val_set_videos/` – Validation set videos
  - `val_json/` – Annotations for the validation set

- `train.zip`, `train.zip.002`, `train.zip.003` – Multi-part archive. When `train.zip` is extracted, it automatically combines the parts to create:
  - `train_set_videos/` – Training set videos
  - `train_json/` – Annotations for the training set

- `train.txt`, `val.txt`, `test.txt` – Text files listing the IDs for each split.

---

## 🛠️ Setup Instructions

### 1. Extract Archives

- Extract `test.zip` → creates `test_set_videos/` and `test_json/`
- Extract `val.zip` → creates `val_set_videos/` and `val_json/`
- Ensure all parts of the training archive are present:
  - `train.zip`, `train.zip.002`, `train.zip.003`
- Extract `train.zip` → automatically reads the parts and creates `train_set_videos/` and `train_json/`

### 2. Organize Split Files

- Create a folder named `txt_files/` in the root directory.
- Move the files `train.txt`, `val.txt`, and `test.txt` into the `txt_files/` folder.

---

## 📁 Final Directory Structure

After completing the extraction and organization steps, your directory should look like this:

---

## 📘 Notes

- The `*_json/` folders contain annotation files corresponding to each dataset split.
- The `txt_files/` folder holds the text files that define the training, validation, and test splits.

---

## 📂 Folder Descriptions

| Folder           | Description                             |
|------------------|-----------------------------------------|
| `train_set_videos/` | Training set video files             |
| `train_json/`        | Training set annotations            |
| `val_set_videos/`   | Validation set video files           |
| `val_json/`         | Validation set annotations           |
| `test_set_videos/`  | Test set video files                 |
| `test_json/`        | Test set annotations                 |
| `txt_files/`        | Split list files (train/val/test)    |
| `1_model_code/`     | Model code directory (if provided)   |

---

## ✅ Setup Complete

Once the above steps are completed, the dataset is ready to be used for training and testing in the **SpEmoC** project.

Dataset : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BQPDFQ


## ✅ Notes:
- Ensure you are in the correct directory before running any script.
- It is recommended to use a virtual environment to manage dependencies cleanly.
