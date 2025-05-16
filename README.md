
# Emotion Recognition Project

This repository contains two main components:

1. **Model Training and Evaluation** (`1_model_code/`)


---
### Contents:
- `train.py`: Script for training the model.
- `test.py`: Script for evaluating the model on the test set.

### Configuration:
- Paths to dataset and pretrained weights are already set within the respective Python files.

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


## 📂 Dataset Structure

- 
- Inside SpEmoC_dataset_submission folder, there are seven subfolders for dataset:
- test.zip - contains test_set_videos ,test_json
- val.zip - contains val_set_videos ,val_json
- train.zip - contains train_set_videos ,train_json
- dataset link : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BQPDFQ
- 
—The train.txt, test.txt, and val.txt files are in the main folder. Create a folder named 'txt_files' and put all three files inside it.

---

## ✅ Notes:
- Ensure you are in the correct directory before running any script.
- It is recommended to use a virtual environment to manage dependencies cleanly.
