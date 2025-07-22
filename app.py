import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace(' ?', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

# Preprocess data
def preprocess(df):
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    X = df.drop("income", axis=1)
    y = df["income"]
    return X, y

X, y = preprocess(df)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Neural Network": MLPClassifier(max_iter=300)
}

# Streamlit UI
st.title("Income Classification (ML Models)")
st.write("Choose a model to train on the Adult Census Dataset")

choice = st.selectbox("Select Algorithm", list(models.keys()))

if st.button("Train and Evaluate"):
    model = models[choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report:")
    st.dataframe(pd.DataFrame(report).transpose())
