# skincancer_detector

# 🧠 Skin Cancer Detection using Deep Learning

A computer vision project to classify skin lesions as **benign** or **malignant (melanoma)** using **Convolutional Neural Networks (CNNs)** and **transfer learning**, trained on the **HAM10000 medical image dataset**.

This project is built and trained in **Google Colab**, with optional deployment using **Streamlit**.

---

## 🔍 Motivation

Skin cancer is one of the most common forms of cancer globally. Early detection of **melanoma** significantly increases survival rates, but visual inspection can be challenging even for experienced dermatologists.

This project explores how **deep learning** can assist in early detection using dermatoscopic images of skin lesions.

---

## 📦 Dataset

- **Source:** [HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- ~10,000 dermatoscopic images labeled with one of 7 diagnoses
- For this project: simplified to **binary classification**
  - `mel`: Melanoma (malignant) → **Label 1**
  - All others (e.g. `nv`, `bkl`, `bcc`, `akiec`, etc.) → **Label 0**


---

## 🚀 How It Works

### ✅ Step-by-step pipeline:

1. **Dataset Preparation**
   - Downloaded and loaded HAM10000 images + metadata
   - Labeled data as binary (melanoma vs. non-melanoma)
   - Balanced the classes using undersampling

2. **Image Preprocessing**
   - Resized images to 224x224
   - Applied data augmentation (rotation, flipping, zoom)
   - Normalized pixel values (0–255 → 0–1)

3. **Model Architecture**
   - Used `MobileNetV2` with pre-trained ImageNet weights
   - Added custom classification head (Dense → Dropout → Dense + Sigmoid)
   - Trained with frozen base model on binary classification

4. **Evaluation**
   - Used confusion matrix, accuracy, precision, recall, F1-score
   - Visualized attention using **Grad-CAM** to see model focus areas

5. **(Optional) Deployment**
   - Simple Streamlit app for uploading images and getting predictions

---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (for evaluation)
- OpenCV (for Grad-CAM)
- Streamlit (for deployment)
- Google Colab (GPU-powered development)

---

## 📊 Results (Sample)

| Metric | Value (example from validation) |
|--------|------------------------------|
| Accuracy | 85% |
| Precision (Melanoma) | 80% |
| Recall (Melanoma) | 75% |
| F1-Score | 77% |

> Note: Actual results may vary based on random train/validation splits.

---

## ✅ To Run This Project

### 🔹 In Google Colab:
1. Clone or upload this notebook and `kaggle.json`
2. Use Kaggle API to download the dataset
3. Follow notebook steps (includes full code and explanations)

### 🔹 To Try Locally:
- Install requirements with `pip install -r requirements.txt`
- Run training script or launch Streamlit app (coming soon)

---

## 🌟 What's Next?

- Unfreeze and fine-tune deeper layers of MobileNetV2
- Handle class imbalance using focal loss or SMOTE
- Expand to multi-class classification (7 skin conditions)
- Deploy with a user-friendly web interface (Streamlit or Gradio)

---

## 🙋‍♂️ Author

Samuel Thomas |Computer Science Student

---

## 📄 License

This project is for educational purposes. Dataset licensed by ISIC Archive under CC BY-NC 4.0.



