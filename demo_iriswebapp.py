import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target != 0) * 1  # Convert to a two-class problem

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Create sliders for the four features
st.sidebar.header('Input Features')
sliders = []
for i, feature in enumerate(iris.feature_names):
    slider = st.sidebar.slider(feature, float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    sliders.append(slider)

# Predict the class label for the input features
features = np.array(sliders).reshape(1, -1)
prediction = model.predict(features)

# Display the predicted label
st.header(f'Predicted label: {iris.target_names[prediction][0]}')
