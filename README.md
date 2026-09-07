# 🧠 NeuroDetect: Brain Tumor & Alzheimer's Stage Classification

## 📌 Overview
NeuroDetect is a deep learning-based medical imaging diagnostic tool designed to classify brain tumors and detect various stages of Alzheimer's disease from MRI scans. Built using a custom Convolutional Neural Network (CNN) architecture with residual connections, this project tackles a complex 7-class neurological dataset. 

To ensure clinical trust and transparency, the pipeline integrates **Explainable AI (Grad-CAM)** to visually highlight the structural regions of the brain that drive the model's predictions.

## ✨ Key Features
- **Dual-Focus Diagnostics:** Capable of classifying 3 types of brain tumors (Glioma, Meningioma, Pituitary) and tracking 4 stages of Alzheimer's progression.
- **Custom CNN Architecture:** Utilizes depthwise separable convolutions and residual blocks for optimized spatial feature extraction and computational efficiency.
- **Robust Training Pipeline:** Overcomes imbalanced medical dataset limitations by computing automated class weights and applying dynamic image augmentations.
- **Explainable AI (XAI):** Generates Grad-CAM heatmaps for test samples, allowing users to see exactly *where* the model is focusing its attention to make a diagnosis.

## 📂 Dataset Structure
The project expects a root `Dataset/` directory containing MRI scans categorized into 7 distinct classes:
- **Brain Tumors:** `Glioma`, `Meningioma`, `Pituitary`
- **Alzheimer's Stages:** `No_Disease`, `Very_Mild_Demented`, `Mild_Demented`, `Moderate_Demented`

## 🚀 Performance & Results
- Achieved an average **F1-score of 86%** for brain tumor detection.
- Utilizes automated learning rate reduction (`ReduceLROnPlateau`) and model checkpointing to preserve the highest-performing weights during training.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Detection-using-CNN.git
   cd Detection-using-CNN
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (Command Prompt)
   python -m venv venv
   venv\Scripts\activate
   
   # Windows (Git Bash)
   python -m venv venv
   source venv/Scripts/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install tensorflow split-folders opencv-python matplotlib scikit-learn numpy
   ```

4. **Prepare the Data:**
   Ensure your 7 medical imaging folders are placed inside a folder named `Dataset/` in the project root.

## 💻 Usage

**To train the model and generate evaluations:**
```bash
python brain_tumor_detection_alz.py
```
The script will automatically:
1. Split the data into Train (70%), Validation (15%), and Test (15%) sets.
2. Train the model across 50 epochs (saving the best weights to `neurodetect_best.keras`).
3. Output a detailed precision/recall classification report.
4. Display a Grad-CAM heatmap for a sample test image.

## 🔬 Roadmap & Future Improvements
- Modularize the codebase into separate `dataset.py`, `model.py`, and `train.py` scripts.
- Transition to a pre-trained backbone (e.g., `EfficientNetB0` or `ResNet50`) using Transfer Learning to improve Alzheimer's classification metrics.
- Upgrade the data loading pipeline from `ImageDataGenerator` to the modern `tf.data` API for faster GPU prefetching.