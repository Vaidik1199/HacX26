# 👁 Eye Disease Detection using Deep Learning

## 📌 Overview

This project is developed as part of **KaggleHacX ’26**, a 6-hour data science hackathon.
The goal is to build a machine learning model that can classify retinal images into different eye disease categories.

The system uses a deep learning model to analyze retinal images and predict the disease type with high accuracy.

---

## 🎯 Problem Statement

Given a retinal (fundus) image, classify it into one of the following categories:

* Normal
* Cataract
* Glaucoma
* Diabetic Retinopathy

---

## 🧠 Model & Approach

* Model Used: EfficientNet-B0 (Transfer Learning)
* Framework: PyTorch
* Training Strategy:

  * Pretrained weights (ImageNet)
  * Fine-tuned final layers
  * Data augmentation applied

---

## 🧹 Data Preprocessing

* Image resizing → 224 × 224
* Normalization
* Augmentation:

  * Horizontal Flip
  * Rotation

---

## 📊 Performance

* Training Accuracy: ~93%
* Validation Accuracy: ~94%

The model shows strong generalization and minimal overfitting.

---

## 🏗 Project Structure

```
HacX26/
│
├── train/                  # Training dataset
├── test/                   # Test images
├── process.ipynb          # Model training notebook
├── model.pth              # Saved trained model
│
└── app/
    └── app.py             # Streamlit demo app
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision streamlit pillow
```

### 2. Run the app

```bash
cd app
streamlit run app.py
```

---

## 💻 Demo Application

The project includes a **Streamlit web app** where users can:

* Upload a retinal image
* Get disease prediction
* View confidence score

---

## 📁 Output

The model generates:

* Predictions for input images
* `solution.csv` file for Kaggle submission

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 👥 Team

* Team Name: *[CyberKnightsz]*
* Members: *[Vaidik Choudhary, Aditya Dhuppe, Ayush Dhule]* 

---

## 🏁 Conclusion

This project demonstrates how deep learning can be applied in medical imaging to assist in early detection of eye diseases. The solution is fast, scalable, and can be extended for real-world healthcare applications.
