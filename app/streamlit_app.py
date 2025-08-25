import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("Breast Cancer Classifier (Demo)")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(Xtr)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)

st.write("Fyll inn noen features (bruk mean-verdier om du er usikker):")
defaults = X.mean()
inputs = []
for col in X.columns[:10]:  # for demo, bare de 10 første
    val = st.number_input(col, float(X[col].min()), float(X[col].max()), float(defaults[col]))
    inputs.append(val)

if st.button("Prediker"):
    vec = np.array(inputs + list(defaults[10:].values)).reshape(1, -1)
    vec = scaler.transform(vec)
    prob = model.predict_proba(vec)[0,1]
    st.metric("Sannsynlighet for BENIGN", f"{prob:.3f}")
    st.write("Prediksjon:", "Benign" if prob>=0.5 else "Malignant")
