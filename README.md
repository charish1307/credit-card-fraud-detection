# 🔍 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AI-powered real-time fraud detection for financial transactions** using Machine Learning and a Streamlit web interface.

---

## 📌 Overview

Credit card fraud costs the global economy billions of dollars each year. This project builds a complete end-to-end ML pipeline that:

- Detects fraudulent credit card transactions with **high accuracy**
- Handles severely **imbalanced data** using SMOTE
- Compares **3 ML models**: Logistic Regression, Random Forest, XGBoost
- Provides a **real-time Streamlit web app** for single & batch predictions

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions**: 284,807 (492 frauds = only 0.172%)
- **Features**: 30 total (Time, Amount + V1-V28 PCA-transformed features)
- **Challenge**: Highly imbalanced — solved using SMOTE oversampling

---

## 🏗️ Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py     # Main ML pipeline (EDA, training, evaluation)
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── .gitignore
│
├── data/                  # Place creditcard.csv here (download from Kaggle)
│   └── creditcard.csv
│
├── models/                # Saved trained models (auto-created)
│   └── best_fraud_detector.pkl
│
├── plots/                 # Visualizations (auto-created)
│   ├── class_distribution.png
│   ├── roc_curves.png
│   ├── confusion_matrix_*.png
│   └── feature_importance_*.png
│
└── model_results.csv      # Model comparison results (auto-created)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/charish1307/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place it in the `data/` folder

---

## 🚀 Usage

### Step 1: Train the Model
```bash
python fraud_detection.py
```
This will:
- Load and explore the data (EDA)
- Apply SMOTE to balance classes
- Train 3 ML models (Logistic Regression, Random Forest, XGBoost)
- Evaluate and compare model performance
- Save the best model to `models/best_fraud_detector.pkl`
- Generate visualizations in `plots/`

### Step 2: Launch the Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🤖 Models & Performance

| Model | ROC-AUC | F1 Score | Notes |
|-------|---------|----------|-------|
| Logistic Regression | ~0.97 | ~0.79 | Fast baseline |
| Random Forest | ~0.99 | ~0.87 | Robust ensemble |
| **XGBoost** | **~0.99** | **~0.89** | **Best performer** |

> Results may vary slightly depending on random state and SMOTE sampling.

---

## 📈 Pipeline Steps

1. **Data Loading** — Read CSV, display shape and stats
2. **EDA** — Class distribution, transaction amounts, correlation heatmap
3. **Preprocessing** — StandardScaler for Amount & Time, SMOTE for imbalance
4. **Model Training** — Logistic Regression, Random Forest, XGBoost
5. **Evaluation** — ROC-AUC, Precision-Recall, F1, Confusion Matrix
6. **Feature Importance** — Top contributing features for tree models
7. **Model Saving** — Best model saved with joblib

---

## 🖥️ Web App Features

| Feature | Description |
|---------|-------------|
| 🔎 Single Transaction | Enter transaction details manually, get instant fraud prediction |
| 📁 Batch Prediction | Upload a CSV file, get fraud predictions for all transactions |
| 📊 Model Info | View dataset info, model comparison, and saved visualizations |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & metrics |
| XGBoost | Gradient boosting |
| imbalanced-learn | SMOTE oversampling |
| Matplotlib & Seaborn | Visualizations |
| Streamlit | Web application |
| Joblib | Model persistence |

---

## 👤 Author

**Charish Yadavali**
- GitHub: [@charish1307](https://github.com/charish1307)
- LinkedIn: [linkedin.com/in/charishyadavali](https://www.linkedin.com/in/charishyadavali)

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ If you found this useful, please star the repo!
