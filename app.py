import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open("email_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Email Spam Detector")
st.write("Upload your email content below, and we will determine if it's spam or not.")

# Text input for email content
email_text = st.text_area("Enter your email content here:")

if st.button("Check for Spam"):
    if email_text:
        # Preprocess and predict
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)
        
        # Display result
        if prediction[0] == 'spam':
            st.error("🚨 This email is classified as SPAM!")
        else:
            st.success("✅ This email is NOT spam.")
    else:
        st.warning("Please enter email content before checking.")
