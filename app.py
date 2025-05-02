
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Title
st.title("Fake News Detection App")
st.subheader("Exposing the truth with NLP and ML")

# Upload CSV
uploaded_file = st.file_uploader("Upload your preprocessed CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing steps (if not already done)
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Check necessary columns
    required_cols = {'Word_count', 'Number_of_words', 'Unique_words', 'Average_Word_Length', 'Label'}
    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain columns: Word_count, Number_of_words, Unique_words, Average_Word_Length, Label")
    else:
        # Define features and label
        X = df[['Word_count', 'Number_of_words', 'Unique_words', 'Average_Word_Length']]
        y = df['Label']

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        st.write("### Model Evaluation:")
        st.text(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        # Prediction section
        st.write("### Try Your Own Prediction")
        wc = st.number_input("Word Count", min_value=0)
        nw = st.number_input("Number of Words", min_value=0)
        uw = st.number_input("Unique Words", min_value=0)
        awl = st.number_input("Average Word Length", min_value=0.0)

        if st.button("Predict"):
            input_data = np.array([[wc, nw, uw, awl]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            result = "Real News" if prediction == 1 else "Fake News"
            st.success(f"Prediction: {result}")
