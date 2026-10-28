# 🎀 Breast Cancer Prediction Studio

An interactive machine learning application that uses Logistic Regression to predict breast cancer classifications from five cell-nucleus measurements, with visual explanations of the model's output.

## 🚀 Live Demo

**[Launch Breast Cancer Prediction Studio](PASTE_YOUR_RENDER_URL_HERE)**

> **Disclaimer:** This project is for educational purposes only. It is not a medical device and must not be used to diagnose or make treatment decisions.

## ✨ Features

- **Interactive predictions:** Adjust five input measurements to explore model predictions.
- **Risk gauge:** Visualize the estimated probability of a malignant classification.
- **Feature contributions:** Explore how individual measurements influence the model output.
- **What-if analysis:** Change one measurement at a time and observe prediction changes.
- **Breast anatomy illustration:** Explore a schematic illustration of breast tissue and a magnified cell nucleus.
- **Decision map:** Visualize the model's decision boundary for two selected features.
- **Data distributions:** Compare input measurements with the dataset distributions.
- **Model explanations:** Review the learned model weights and limitations.

## 🧠 Model

- Algorithm: Logistic Regression
- Input features: Mean radius, mean texture, mean perimeter, mean area, and mean smoothness
- Preprocessing: Feature scaling using the saved scaler
- Output: Estimated malignant and benign class probabilities

## 🛠️ Tech Stack

- Python
- Gradio
- scikit-learn
- NumPy and pandas
- Matplotlib and SciPy
- Joblib
- Render for deployment

## 📁 Project Files

- `app.py` — Gradio application
- `easy_model.joblib` — Saved scaler and trained model
- `Breast_cancer_data.csv` — Dataset used for visualizations
- `requirements.txt` — Python dependencies

## ⚠️ Limitations

This educational demonstration uses five summary features and a Logistic Regression model. Its predictions are not a substitute for clinical assessment, medical imaging, biopsy, or professional medical advice.
