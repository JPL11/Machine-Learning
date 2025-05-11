# 🧠 MRI Brain Tumor Detection with Physics-Informed Denoising and YOLOv12

This repository presents a **novel brain tumor detection pipeline** that integrates a **Physics-Informed Denoising Module (MRIPHYS-Net)** with the cutting-edge **YOLOv12 object detection framework**. The proposed approach enhances tumor detection accuracy under challenging noise conditions by combining MRI acquisition physics, adaptive noise estimation, and contrast enhancement directly into the pipeline.

---

## 📚 Key Contributions

- ✅ **Physics-Informed Denoising Module (MRIPHYS-Net)**
  - Integrates k-space domain knowledge directly into the denoising process.
  - Effectively addresses Rician noise common in MRI imaging.

- ✅ **Adaptive Noise Estimation**
  - Dynamically estimates noise patterns during inference.
  - Removes reliance on fixed noise assumptions.

- ✅ **Seamless YOLOv12 Integration**
  - Directly integrates the denoising module before YOLOv12 inference.
  - Optimizes end-to-end tumor detection accuracy.

- ✅ **Contrast Enhancement for Tumor Visibility**
  - Highlights tumor regions based on domain knowledge to improve detectability.

- ✅ **Comprehensive Experimental Results**
  - Evaluated across multiple noise levels.
  - Compared against baseline YOLO models and standard denoisers.

[📂 **View Results and Training Visualizations**](https://drive.google.com/drive/u/1/folders/1rPyKZ-rSk19twk1szAPLNyJIJvtLtgnj)

---

## 📂 Dataset Details

- **Source:** [Roboflow Labeled MRI Brain Tumor Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Resolution:** Varies, preprocessed to **640×640** for YOLOv12
- **Dataset Size:**
  - Training Images: **1695**
  - Validation Images: **502**
  - Total Images: **2197**
- **Input Format:** JPEG
- **Label Format:** YOLOv5/YOLOv12 format (`class x_center y_center width height`)

---

## 📊 Performance Metrics

| **Model**                      | **mAP@0.5** | **mAP@0.5:0.95** | **Precision** | **Recall** | **FPS** |
|-------------------------------|-------------|------------------|---------------|------------|---------|
| YOLOv5 Baseline               | 94.7%       | 56.8%            | 92.5%         | 91.6%      | 128     |
| YOLOv11 Baseline              | 96.9%       | 61.6%            | 94.1%         | 92.8%      | 172     |
| YOLOv11 + Denoising           | 95.9%       | 60.3%            | 92.4%         | 93.8%      | 357     |
| YOLOv12 Baseline              | 96.9%       | 61.2%            | 92.9%         | 95.0%      | 476     |
| YOLOv12 + Previous Denoiser   | 96.1%       | 60.3%            | 90.2%         | 93.3%      | 244     |
| YOLOv12 + New Denoiser        | 96.1%       | 60.3%            | 90.2%         | 93.4%      | 357     |
| **YOLOv12 + MRIPHYS-Net**     | **96.1%**   | **60.3%**        | **90.2%**     | **93.4%**  | **323** |

---
