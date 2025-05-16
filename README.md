
# 🧠 Handwritten Math Symbol Detector

This project is a deep learning-based classifier that detects and classifies **82 different handwritten mathematical symbols** using a **Convolutional Neural Network (CNN)**.


---

## 🖼️ Project Overview

The goal of this project is to recognize a wide variety of handwritten math symbols from images using deep learning. This model can be used in OCR applications, education tools, and math-solving interfaces.

---

## 📂 Dataset

- 📦 Dataset Source: [Handwritten Math Symbols on Kaggle](https://www.kaggle.com/your-dataset-link)
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

Saved as: `model/finalmodel.h5`

---

## 📈 Performance

- The model achieves high training and validation accuracy
- Capable of recognizing diverse symbol types

---

## 📊 Visualization

- Visualized training curves using Matplotlib
- Data distribution and sample predictions plotted with Seaborn and Matplotlib

---

## 🧪 Research & Documentation

- 📄 [Research Paper PDF](./docs/math_symbol_detection_paper.pdf): Overview of data, model, experiments, and results
- 🧠 [Trained Model](./model/finalmodel.h5): Ready for prediction and deployment

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

**Author**: [Your Name]  
**GitHub**: [https://github.com/your-username](https://github.com/your-username)
