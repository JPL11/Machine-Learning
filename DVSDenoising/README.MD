# Event Camera Background Activity Denoising (MLPF Filter Project)

## Project Overview and Goals

Event cameras (Dynamic Vision Sensors) output asynchronous event streams rather than traditional frames. A major challenge in using DVS data is **background activity (BA) noise**—spurious events caused by thermal effects, circuit leakage, or low-light noise. This project replicates and extends the "Low Cost and Latency Event Camera Background Activity Denoising" (T-PAMI 2023) paper by Shasha Guo and Tobi Delbruck.

### Project Goals
- Implement and evaluate three denoising filters: BAF, STCF, and MLPF.
- Train a lightweight Multi-Layer Perceptron Filter (MLPF) on labeled event patches.
- Provide Colab notebooks for training, evaluation, and deployment.
- Convert trained models into `.pb` format for real-time use in jAER.
- Compare our MLPF against the authors' published models using ROC/AUC.

---

## 📦 Datasets

### 1. HotelBar Dataset
- Source: DND21 Project  
- Download: [https://sites.google.com/view/dnd21](https://sites.google.com/view/dnd21)

### 2. Dynamic 6DOF Dataset (Moving Camera)
- Source: UZH-RPG  
- Download: [https://rpg.ifi.uzh.ch/davis_data.html](https://rpg.ifi.uzh.ch/davis_data.html)

### 3. Mixed Labeled CSV Data (Used for Training)
- Google Drive: [Training Data CSVs](https://drive.google.com/drive/folders/1fR8x3lTcO7qpxfItNjvAbTx_6hyiI6i9)

Each row in the CSV contains a timestamp, label (1 = signal, 0 = noise), and a flattened 7x7 or 25x25 time-surface patch.

---

## ⚙️ Setup & Dependencies

### Requirements
- Python 3.9+
- TensorFlow 2.8–2.13

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```bash

## 🚀 How to Train the MLPF Model
1. Open DVSNetWorkDenoiser.ipynb in Google Colab

2. Load the training CSV dataset (see Datasets section)

3. Configure patch size (e.g., 7x7)

4. Train a model:

- Architecture: [147 → Dense(20, relu) → Dropout(0.2) → Dense(1, sigmoid)]

- Optimizer: Adam (lr=1e-3), Loss: MSE, Epochs: 5

5. Save as: mlpf_model_best.h5

##🧪 Evaluation: ROC and AUC
Filter	HotelBar AUC	Driving AUC
BAF	~0.89	~0.79
STCF	~0.96	~0.83
MLPF	~0.97	~0.89

## 🧠 Deploying in jAER
1. Convert .h5 model to .pb (frozen graph)

2. Load .pb into jAER's MLPNoiseFilter GUI

3. Adjust threshold (default: 0.5)

4. Ensure your .pb model has:

- Input node: input

- Output node: output

- GraphDef version < 1994 for full compatibility
