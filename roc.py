import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_data(n_samples, n_features, random_state):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2,
                               n_clusters_per_class=1, random_state=random_state)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df['Target'] = y
    df.to_csv("data.csv", index=False)


    return X, y

# Function to train logistic regression model
def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Function to calculate TPR and FPR for a given threshold
def calculate_tpr_fpr(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot()

# Main Streamlit app
st.title("Logistic Regression - ROC Curve and Confusion Matrix")

# Adjustable parameters
threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=500)
n_features = st.sidebar.slider("Number of Features", min_value=2, max_value=10, value=5)
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.3, step=0.1)
random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42)

# Generate synthetic data
X, y = generate_data(n_samples, n_features, random_state)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train logistic regression model
lr_model = train_logistic_regression(X_train, y_train)

# Predict probabilities on test set
y_pred_proba = lr_model.predict_proba(X_test)

# Calculate TPR and FPR for the selected threshold
tpr, fpr = calculate_tpr_fpr(y_test, y_pred_proba, threshold)

# Calculate AUC
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba[:, 1])
auc_score = auc(fpr_curve, tpr_curve)

# Display ROC Curve
plot_roc_curve(fpr_curve, tpr_curve, auc_score)

# Display AUC
st.write(f"AUC (Area Under Curve): {auc_score:.2f}")

# Display confusion matrix
st.subheader("Confusion Matrix")
confusion = confusion_matrix(y_test, lr_model.predict(X_test))
st.write(pd.DataFrame(confusion, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive']))

# Display TPR and FPR for the selected threshold in a styled box
st.subheader("TPR and FPR for Selected Threshold")
st.info(f"**True Positive Rate (TPR):** {tpr:.2f}")
st.info(f"**False Positive Rate (FPR):** {fpr:.2f}")



f1 = f1_score(y_test, lr_model.predict(X_test))
st.info(f"F1 Score: {f1:.2f}")