
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


The dataset for SpEmoC is organized inside the SpEmoC_dataset_submission folder. Follow the steps below to correctly set up the dataset:

📦 **Contents of** SpEmoC_dataset_submission

train.zip, train.zip.002, train.zip.003 – multi-part archive containing:train_set_videos/ – video clips for the training set
                                                                        train_json/ – annotations for the training set
test.zip – contains:test_set_videos/ – video clips for the test set
                    test_json/ – annotations for the test set

val.zip – contains:val_set_videos/ – video clips for the validation set
                  val_json/ – annotations for the validation set

train.txt, val.txt, test.txt – text files listing the video IDs for each split

🛠️ **Setup Instructions**
**1.Unzip the archives:**

   Extract test.zip – this will create test_set_videos/ and test_json/.
   Extract val.zip – this will create val_set_videos/ and val_json/.
   Ensure all parts of the train archive are present (train.zip, train.zip.002, train.zip.003) and extract train.zip. It will automatically combine the parts and create train_set_videos/ and 
   train_json/.

**2.Organize text files:**

   Create a new folder named txt_files/ in the main directory.
   Move the three text files train.txt, val.txt, and test.txt into this folder.

**3.Final Folder Structure:**

After extraction and organization, your directory structure should look like this:

SpEmoC_dataset_submission/
├── 1_model_code/
├── train_set_videos/
├── train_json/
├── val_set_videos/
├── val_json/
├── test_set_videos/
├── test_json/
├── txt_files/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt

📝 Note:
The *_json/ folders contain annotation files for the respective dataset splits.

## ✅ Notes:
- Ensure you are in the correct directory before running any script.
- It is recommended to use a virtual environment to manage dependencies cleanly.
