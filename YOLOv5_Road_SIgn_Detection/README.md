# YOLOv5 Road Sign Detection with Custom Architecture Modifications

This repository demonstrates an object detection pipeline for road sign detection using a modified YOLOv5 architecture. The project introduces key architectural changes, making the YOLOv5 model more efficient and effective for this task. The repository includes scripts, notebooks, and configuration files for training and evaluating the modified YOLOv5 on two different datasets.

---

## 📂 Repository Structure
├── notebooks/
│   ├── dataset1_training.ipynb      # Notebook for training on the first dataset
│   ├── dataset2_training.ipynb      # Notebook for training on the second dataset
├── models/
│   ├── common.py                    # Modified YOLOv5 modules (SE Block and BiFPN)
│   ├── yolo.py                      # Detection-specific modules
│   └── yolov5s.yaml         # Model architecture definition
├── dataset/
│   └── data.yaml                    # Update with paths to your dataset
└── README.md                        # Project documentation

---

## ✨ Modifications to YOLOv5 Architecture

This project incorporates two significant architectural changes into the YOLOv5 model:

1. **Bi-Directional Feature Pyramid Network (BiFPN)**:
   - BiFPN replaces the original YOLOv5 neck. It enables bidirectional feature fusion and incorporates learnable attention weights to prioritize meaningful features at different scales.
   - Implementation can be found in `models/common.py`.

2. **Squeeze-and-Excitation (SE) Blocks**:
   - SE Blocks introduce channel-wise attention, allowing the model to focus on the most informative channels in feature maps.
   - These blocks have been added to the backbone for enhanced feature extraction. See `models/common.py` for the implementation.

### Modified Configuration
The model architecture is defined in `models/yolov5s.yaml`. The changes include:
- **Backbone**: Incorporation of SE blocks to enhance feature extraction.
- **Neck**: Replaced the standard FPN with BiFPN for improved multi-scale feature fusion.

---

## 🏗️ How to Use

### 1. Setup
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
### 2. Dataset Preparation
	•	Prepare your dataset in YOLOv5 format (with images and labels).
	•	Update dataset/data.yaml:
	•	Set train and val to the paths of your training and validation datasets.
	•	Specify the number of classes and their names.

### 3. Training

Choose the notebook for the appropriate dataset:
	•	For Dataset 1: Use YOLOv5proposed1.ipynb.
	•	For Dataset 2: Use YOLOv5proposed2.ipynb.

Modify the data.yaml file in the dataset/ folder to reflect the dataset configuration before running the notebook.

### 4. Evaluation

Evaluate the model using the final trained weights. Save the results for comparison and analysis.
