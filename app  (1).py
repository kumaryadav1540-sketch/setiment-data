import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")

st.title("Sentiment Analysis App")

text = st.text_input("Enter any sentence")

if st.button("Predict"):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == "Positive":
        st.success("😊 Positive")
    elif prediction[0] == "Negative":
        st.error("☹️ Negative")
    else:
        st.warning("😐 Neutral")