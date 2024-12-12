# YOLOv5 Road Sign Detection with Custom Architecture Modifications

This repository demonstrates an object detection pipeline for road sign detection using a modified YOLOv5 architecture. The project introduces key architectural changes, making the YOLOv5 model more efficient and effective for this task. The repository includes scripts, notebooks, and configuration files for training and evaluating the modified YOLOv5 on two different datasets.

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

## 🛠️ Datasets

This project uses two datasets for training and evaluation:

1. **[Korean Traffic Signs](https://universe.roboflow.com/work-9vtar/traffic-sign-detection-yolov5/dataset/1)**:
   - Dataset with class labels represented as numbers.
   - Contains only Korean traffic signs.

2. **[Mixed Traffic Signs (Korean and English)](https://universe.roboflow.com/andres-calderon-unab/traffic-signs-dsjv6/dataset/3)**:
   - Dataset with both Korean and English traffic signs.
   - Class labels are written out (e.g., "Speed Limit" instead of numeric).

#### Instructions:
Both datasets are available in multiple formats to suit different architectures and frameworks.
Press the **Download** button for the desired format, such as **YOLOv5**, to begin.
Ensure that your datasets are downloaded and prepared in YOLOv5 format (with images and labels). Update the `data.yaml` file in the `dataset/` folder to reflect the dataset paths before training.

---

## 🏗️ How to Use

### 1. Setup
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
### 2. Replace Files

Replace the original files in the models/ directory of the YOLOv5 repository with the modified files provided in this project:

	•	Replace models/common.py with the version in this repository.
	•	Replace models/yolo.py with the version in this repository.
	•	Replace the yolov5_modified.yaml in the models/ directory.

The modified files include the implementations of the SE Block, BiFPN, and other architectural changes.

### 3. Running the Notebooks in Google Colab

The provided notebooks are designed to be executed in Google Colab to leverage their GPU resources for training.

	1.	Open the desired notebook in Google Colab:
 
	•	For Dataset 1 (Korean Traffic Signs): Use YOLOv5proposed1.ipynb.
	•	For Dataset 2 (Mixed Traffic Signs): Use YOLOv5proposed2.ipynb.
 
	2.	Upload the data.yaml file to Colab:
 
 	•	Prepare your dataset in YOLOv5 format (with images and labels).
	•	Update dataset/data.yaml:
	•	Set test, train and val to the paths of your testing, training and validation datasets.
	•	Specify the number of classes and their names.
 
	3.	Enable GPU acceleration:
 
	•	Go to Runtime > Change runtime type > Hardware accelerator > GPU.
 
	4.	Execute the cells to train the model.

### 4. Training

Choose the notebook for the appropriate dataset:

	•	For Dataset 1: Use YOLOv5proposed1.ipynb. 
	•	For Dataset 2: Use YOLOv5proposed2.ipynb.

Modify the data.yaml file in the dataset/ folder to reflect the dataset configuration before running the notebook.

### 5. Evaluation

Evaluate the model using the final trained weights. Save the results for comparison and analysis.

---

## 📊 Results

The proposed architecture and model were compared with baseline architectures of YOLOv5 and YOLOv7 using the first dataset. The baseline code for these comparisons is not provided here, as it uses the standard implementations available in the respective repositories. The modifications in this repository focus on improving feature fusion and attention mechanisms over the standard baselines.

YOLOv5 baseline

![33fc872617042166ba3c4ba5c08b39c7fca32b7ba4ab9d917773f60df7f73720](https://github.com/user-attachments/assets/efe94943-28dd-4678-8d68-84a411187ff6)

YOLOv7 baseline

![Screenshot 2024-12-11 at 2 03 08 PM](https://github.com/user-attachments/assets/3d2fdb72-f1e8-47d7-b7ce-0dbf5d9c2311)

YOLOv5 proposed

![d36c4870f6fa62f71b53bac59e2b809b8866a7e9be89a057fd3c6ef764bb0277](https://github.com/user-attachments/assets/bae09ea1-80de-4137-8f74-5aee15fac049)


---

## 🛠️ Code Highlights
	•	models/common.py: Contains the implementation of SE blocks and BiFPN.
	•	models/yolo.py: Integrates the modified layers into the detection pipeline.
	•	models/yolov5_modified.yaml: Defines the modified architecture for YOLOv5.

---

## 🔧 Future Work
	•	Explore further enhancements to the architecture.
	•	Experiment with additional datasets and augmentations.
	•	Optimize training parameters for real-time applications.

---
