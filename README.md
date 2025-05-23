
# 🧠 Handwritten Math Symbol Detector

This project is a deep learning-based classifier that detects and classifies **82 different handwritten mathematical symbols** using a **Convolutional Neural Network (CNN)**.


---

## 🖼️ Project Overview

The goal of this project is to recognize a wide variety of handwritten math symbols from images using deep learning. This model can be used in OCR applications, education tools, and math-solving interfaces.
![image alt](https://github.com/Harsh3202/Handwritten-equation-solver/blob/f9cdad47d6f2fdfddba78c71b7162aed5457e26c/Screenshot%202025-05-16%20183549.png)

---

## 📂 Dataset

- 📦 Dataset Source: [Handwritten Math Symbols on Kaggle](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)
- 82 classes of handwritten mathematical symbols
- Images were extracted from a `.rar` archive and loaded using TensorFlow's `image_dataset_from_directory`.

---

## 🧹 Data Preprocessing

- Resized all images to 128×128 pixels
- Converted to grayscale and normalized to range [0, 1]
- Used 80% of data for training, 10% for validation, 10% for testing
- Preprocessed using TensorFlow utilities with batching and shuffling

---

## 🧠 Model Architecture

Built using TensorFlow and Keras:

- `3 × Conv2D` layers (ReLU)
- `3 × MaxPooling2D` layers
- `1 × Flatten` layer
- `3 × Dense` hidden layers
- `2 × Dropout` layers for regularization
- `1 × Dense` output layer (softmax activation for 82 classes)

Saved as: `finalmodel.h5`
Download it from the link given below

---

## 📈 Performance

- The model achieves high training and validation accuracy
- Capable of recognizing diverse symbol types

---

## 📊 Visualization

- Visualized training curves using Matplotlib
- Data distribution and sample predictions plotted with Seaborn and Matplotlib
![image alt](https://github.com/Harsh3202/Handwritten-equation-solver/blob/f9cdad47d6f2fdfddba78c71b7162aed5457e26c/Screenshot%202025-05-16%20183619.png)

---

## 🧪 Research & Documentation

- 📄 [Research Paper PDF](https://github.com/Harsh3202/Handwritten-equation-solver/blob/f9cdad47d6f2fdfddba78c71b7162aed5457e26c/IEE%7B1%7D.pdf): Overview of data, model, experiments, and results
- 🧠 [Trained Model](https://drive.google.com/file/d/1-4UVlHxI5rW9y_s6JV0MMV_QsEx-w8qI/view?usp=sharing): Ready for prediction and deployment

---

## 🛠️ Libraries Used

- TensorFlow
- Keras
- Matplotlib
- Seaborn
- NumPy

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run notebook:
   ```bash
   jupyter notebook maths_symbol_prediction_model.ipynb
   ```

---

## 📌 Future Improvements

- Add full equation detection from image
- Deploy real-time web interface using Gradio
- Expand dataset and enhance class labels

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
