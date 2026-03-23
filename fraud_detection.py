# =============================================================
# Credit Card Fraud Detection - ML Pipeline
# Author: Charish Yadavali
# Dataset: Kaggle Credit Card Fraud Detection
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os


def load_data(filepath='data/creditcard.csv'):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df


def perform_eda(df):
    print("\n=== EDA ===")
    print(df.describe())
    fraud_count = df['Class'].value_counts()
    print(f"\nClass distribution:\n{fraud_count}")
    print(f"Fraud percentage: {fraud_count[1]/len(df)*100:.4f}%")

    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df, palette=['steelblue', 'crimson'])
    plt.title('Class Distribution (0=Normal, 1=Fraud)')
    plt.savefig('plots/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    df[df['Class'] == 0]['Amount'].hist(bins=50, color='steelblue', alpha=0.7)
    plt.title('Normal Transaction Amounts')
    plt.subplot(1, 2, 2)
    df[df['Class'] == 1]['Amount'].hist(bins=50, color='crimson', alpha=0.7)
    plt.title('Fraudulent Transaction Amounts')
    plt.tight_layout()
    plt.savefig('plots/amount_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("EDA plots saved to 'plots/' folder.")


def preprocess(df):
    print("\n=== Preprocessing ===")
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("Applying SMOTE to handle class imbalance...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE - Training set: {X_train_res.shape}")
    return X_train_res, X_test, y_train_res, y_test


def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
    }

    trained_models = {}
    print("\n=== Training Models ===")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} - Done.")
    return trained_models


def evaluate_models(models, X_test, y_test):
    print("\n=== Model Evaluation ===")
    results = {}
    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(10, 7))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'ROC-AUC': round(roc_auc, 4),
            'Avg Precision': round(avg_prec, 4),
            'F1 Score': round(f1, 4)
        }

        print(f"\n--- {name} ---")
        print(f"ROC-AUC: {roc_auc:.4f} | Avg Precision: {avg_prec:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"plots/confusion_matrix_{name.replace(' ', '_')}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})')

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend()
    plt.savefig('plots/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n=== Summary ===")
    results_df = pd.DataFrame(results).T
    print(results_df)
    results_df.to_csv('model_results.csv')
    return results


def plot_feature_importance(models, feature_names):
    os.makedirs('plots', exist_ok=True)
    for name in ['Random Forest', 'XGBoost']:
        if name in models:
            model = models[name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(indices)), importances[indices], color='steelblue')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.title(f'Top 20 Feature Importances - {name}')
            plt.tight_layout()
            plt.savefig(f"plots/feature_importance_{name.replace(' ', '_')}.png",
                        dpi=150, bbox_inches='tight')
            plt.close()


def save_best_model(models, results):
    os.makedirs('models', exist_ok=True)
    best_name = max(results, key=lambda x: results[x]['ROC-AUC'])
    joblib.dump(models[best_name], 'models/best_fraud_detector.pkl')
    print(f"\nBest model: {best_name} (ROC-AUC: {results[best_name]['ROC-AUC']})")
    print("Saved to models/best_fraud_detector.pkl")
    return best_name


if __name__ == '__main__':
    df = load_data('data/creditcard.csv')
    perform_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    feature_names = list(X_test.columns)
    plot_feature_importance(models, feature_names)
    best = save_best_model(models, results)
    print(f"\nPipeline complete! Best model: {best}")
