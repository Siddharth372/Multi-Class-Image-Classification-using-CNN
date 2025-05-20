# Multi-Class-Image-Classification-using-CNN

This project performs image classification on the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset using Convolutional Neural Networks (CNNs), transfer learning with VGG16, and ensemble learning techniques.

---

## 📂 Dataset

- **Source**: [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Classes**:
  - `buildings`
  - `forest`
  - `glacier`
  - `mountain`
  - `sea`
  - `street`
- **Structure**:
  
 ├── seg_train/
 │ ├── buildings/
 │ ├── forest/
 │ └── ...
 ├── seg_test/
 ├── seg_pred/

---

## 🚀 Features

- 📷 Image classification using a custom-built CNN
- 🧠 Transfer learning with pre-trained VGG16
- 🔁 Ensemble learning using bagging
- 📉 PCA visualization of high-dimensional feature space
- 📊 Performance metrics: Accuracy, loss curves, confusion matrix
- ✅ Final accuracy improvements via fine-tuning

---

## 📌 Project Structure

- `intel_image_classification_cnn_keras_majorProject.ipynb`: Full implementation notebook
- `README.md`: Project overview
- `requirements.txt`: Python dependencies (if provided)

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- tqdm

---

## 📈 Results

- ✅ Base CNN accuracy: ~75%
- ✅ VGG16 features + shallow model: ~84%
- ✅ Ensemble learning improved stability
- ✅ Fine-tuned VGG16 model delivered the best results

---

## 📝 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
2. Place the `seg_train`, `seg_test`, and optionally `seg_pred` folders in your working directory.
3. Open and run the notebook:
 ```bash
 jupyter notebook intel_image_classification_cnn_keras_majorProject.ipynb
