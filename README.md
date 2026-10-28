# 🎀 Breast Cancer Prediction Studio

An interactive machine learning web application that uses Logistic Regression to classify breast cancer patterns from five cell-nucleus measurements and visualize the model's predictions.

## 🚀 Live Demo

### [**Launch Breast Cancer Prediction Studio**](https://breast-cancer-detection-am48.onrender.com/)

## ✨ Features

- **Interactive predictions** using five input measurements.
- **Risk gauge** displaying the estimated probability of a malignant classification.
- **Feature contribution analysis** to explore how measurements influence predictions.
- **What-if analysis** to observe prediction changes when one feature is adjusted.
- **Schematic breast anatomy visualization** with a magnified cell nucleus.
- **Decision map** showing the model's decision boundary for two features.
- **Feature distributions** comparing measurements with the dataset.
- **Model explanations** displaying learned weights and limitations.

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Input features: Mean radius, mean texture, mean perimeter, mean area, and mean smoothness
- Preprocessing: Feature scaling
- Output: Estimated malignant and benign class probabilities

## 🛠️ Tech Stack

- Python
- Gradio
- scikit-learn
- NumPy and pandas
- Matplotlib and SciPy
- Joblib
- Render

## 📁 Project Structure

- `app.py` — Gradio web application
- `easy_model.joblib` — Saved scaler and trained model
- `Breast_cancer_data.csv` — Dataset for visualizations
- `requirements.txt` — Python dependencies

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions. Predictions are not a substitute for evaluation by a qualified healthcare professional.
