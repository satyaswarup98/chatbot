import streamlit as st
import joblib
import re
import string

# Load the trained model
model = joblib.load("nlp_feedback_classifier.pkl")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.write("Enter customer feedback to classify it as Complaint, Suggestion, or Praise.")

user_input = st.text_input("Enter customer feedback:")

if user_input:
    cleaned_input = preprocess_text(user_input)
    prediction = model.predict([cleaned_input])[0]
    st.write(f"**Feedback category:** {prediction}")
