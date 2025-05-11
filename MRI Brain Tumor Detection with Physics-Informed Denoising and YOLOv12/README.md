🧠 MRI Brain Tumor Detection with Physics-Informed Denoising and YOLOv12

This repository presents a novel brain tumor detection pipeline that integrates a physics-informed denoising module with the cutting-edge YOLOv12 object detection framework. The proposed approach significantly enhances tumor detection performance, particularly under challenging noise conditions, by incorporating MRI acquisition physics, adaptive noise estimation, and task-specific contrast enhancement directly into the pipeline.

⸻

📚 Key Contributions
	•	✅ Physics-Informed Denoising Module
	•	Incorporates k-space domain understanding directly into the denoising process.
	•	Addresses Rician noise common in MRI imaging, improving robustness against high-noise scenarios.
	•	✅ Adaptive Noise Estimation
	•	Dynamically estimates noise patterns during inference, removing the need for fixed noise assumptions.
	•	✅ Seamless YOLOv12 Integration
	•	Integrates the denoising pipeline directly before YOLOv12 inference, optimizing detection accuracy.
	•	✅ Contrast Enhancement for Tumor Visibility
	•	Enhances tumor regions based on domain knowledge to improve detection accuracy.
	•	✅ Comprehensive Experimental Results
	•	Evaluated across multiple noise levels and compared against baseline YOLO models and standard denoisers.
