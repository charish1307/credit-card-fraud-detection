# =============================================================
# Streamlit Web App - Credit Card Fraud Detector
# Run: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Credit Card Fraud Detection System")
st.markdown("**AI-powered real-time fraud detection for financial transactions**")
st.markdown("---")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses Machine Learning to detect fraudulent credit card transactions.\n\n"
    "Models trained on the Kaggle Credit Card Fraud Detection dataset with 284,807 transactions."
)
st.sidebar.markdown("### Tech Stack")
st.sidebar.markdown("- Python, Scikit-learn")
st.sidebar.markdown("- XGBoost, Random Forest")
st.sidebar.markdown("- SMOTE for class balancing")
st.sidebar.markdown("- Streamlit for UI")

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/best_fraud_detector.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Single Transaction", "📁 Batch Prediction", "📊 Model Info"])

# ---------------------------------------------------------------
# TAB 1: Single Transaction Prediction
# ---------------------------------------------------------------
with tab1:
    st.header("Predict a Single Transaction")
    st.markdown("Enter transaction details below. Features V1-V28 are PCA-transformed (anonymized).")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
        time = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)

    st.markdown("**PCA Features (V1 - V28):**")
    v_features = {}
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

    if st.button("🔍 Detect Fraud", use_container_width=True):
        if model is None:
            st.warning("Model not loaded. Please run fraud_detection.py first to train and save the model.")
        else:
            # Preprocess
            scaler = StandardScaler()
            input_data = {**v_features}
            input_data['scaled_amount'] = (amount - 88.35) / 250.12
            input_data['scaled_time'] = (time - 94813.86) / 47488.14
            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚨 FRAUD DETECTED! Confidence: {probability*100:.2f}%")
                st.metric("Fraud Probability", f"{probability*100:.2f}%", delta="High Risk")
            else:
                st.success(f"✅ LEGITIMATE Transaction. Fraud Probability: {probability*100:.2f}%")
                st.metric("Fraud Probability", f"{probability*100:.2f}%", delta="Low Risk")

            # Risk gauge
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh(0, probability, color='crimson' if probability > 0.5 else 'steelblue', height=0.5)
            ax.barh(0, 1-probability, left=probability, color='#e0e0e0', height=0.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Fraud Probability')
            ax.set_title('Risk Score')
            st.pyplot(fig)

# ---------------------------------------------------------------
# TAB 2: Batch Prediction
# ---------------------------------------------------------------
with tab2:
    st.header("Batch Prediction from CSV")
    st.markdown("Upload a CSV file with transaction data. Expected columns: Time, V1-V28, Amount")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} transactions.")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            if model is None:
                st.warning("Model not loaded. Please train the model first.")
            else:
                scaler = StandardScaler()
                df_proc = df.copy()
                if 'Amount' in df_proc.columns:
                    df_proc['scaled_amount'] = scaler.fit_transform(df_proc[['Amount']])
                    df_proc.drop('Amount', axis=1, inplace=True)
                if 'Time' in df_proc.columns:
                    df_proc['scaled_time'] = scaler.fit_transform(df_proc[['Time']])
                    df_proc.drop('Time', axis=1, inplace=True)
                if 'Class' in df_proc.columns:
                    df_proc.drop('Class', axis=1, inplace=True)

                predictions = model.predict(df_proc)
                probabilities = model.predict_proba(df_proc)[:, 1]

                results = df.copy()
                results['Fraud_Prediction'] = predictions
                results['Fraud_Probability'] = probabilities.round(4)
                results['Risk_Level'] = results['Fraud_Probability'].apply(
                    lambda x: 'HIGH' if x > 0.7 else ('MEDIUM' if x > 0.3 else 'LOW')
                )

                fraud_count = predictions.sum()
                st.metric("Total Transactions", len(predictions))
                col1, col2, col3 = st.columns(3)
                col1.metric("Fraudulent", int(fraud_count), delta=f"{fraud_count/len(predictions)*100:.2f}%")
                col2.metric("Legitimate", int(len(predictions)-fraud_count))
                col3.metric("Fraud Rate", f"{fraud_count/len(predictions)*100:.2f}%")

                st.dataframe(results.head(50))
                csv = results.to_csv(index=False)
                st.download_button("Download Results CSV", csv, "fraud_predictions.csv", "text/csv")

# ---------------------------------------------------------------
# TAB 3: Model Info
# ---------------------------------------------------------------
with tab3:
    st.header("Model Information")
    st.markdown("""
    ### Dataset
    - **Source**: Kaggle Credit Card Fraud Detection Dataset
    - **Transactions**: 284,807 (492 frauds = 0.172%)
    - **Features**: 30 (Time, Amount, V1-V28 PCA features)
    - **Class Imbalance**: Handled with SMOTE

    ### Models Trained
    | Model | Description |
    |-------|-------------|
    | Logistic Regression | Baseline linear model |
    | Random Forest | Ensemble of 100 decision trees |
    | XGBoost | Gradient boosting (best performer) |

    ### Evaluation Metrics
    - ROC-AUC Score
    - Average Precision Score
    - F1 Score
    - Confusion Matrix

    ### How to Use
    1. Train the model: `python fraud_detection.py`
    2. Launch this app: `streamlit run app.py`
    3. Use Single Transaction tab for manual input
    4. Use Batch Prediction tab to upload a CSV file
    """)

    if os.path.exists('model_results.csv'):
        st.subheader("Model Comparison Results")
        results_df = pd.read_csv('model_results.csv', index_col=0)
        st.dataframe(results_df)

    for plot_file in ['plots/roc_curves.png', 'plots/class_distribution.png']:
        if os.path.exists(plot_file):
            st.image(plot_file, caption=plot_file.split('/')[-1].replace('.png', '').replace('_', ' ').title())
