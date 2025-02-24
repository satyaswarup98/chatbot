import streamlit as st
import joblib
import pandas as pd
import re
import string
import datetime

# Load the trained model
model = joblib.load("nlp_feedback_classifier.pkl")

# Load dataset for query-based analysis
df = pd.read_csv("Customer Feedback Analysis.csv")

# Preprocess dataset
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['Cleaned_Reviews'] = df['Reviews'].astype(str).apply(preprocess_text)

# Assign labels
def assign_label(rating):
    if rating <= 2:
        return 'Complaint'
    elif rating == 3 or rating == 4:
        return 'Suggestion'
    else:
        return 'Praise'

df['Category'] = df['Rating'].apply(assign_label)

# Simulating date column (assuming last 7 days for query-based filtering)
df['Date'] = pd.date_range(end=datetime.datetime.today(), periods=len(df), freq='D')

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.write("Ask a question about customer feedback.")

# User Input
query = st.text_input("Enter your query:")

# Response Logic
if query:
    if "category" in query.lower():
        user_feedback = st.text_input("Enter customer feedback:")
        if user_feedback:
            cleaned_input = preprocess_text(user_feedback)
            prediction = model.predict([cleaned_input])[0]
            st.write(f"**Feedback category:** {prediction}")

    elif "how many complaints" in query.lower() and "last week" in query.lower():
        last_week = datetime.datetime.today() - datetime.timedelta(days=7)
        complaints_count = df[(df['Category'] == 'Complaint') & (df['Date'] >= last_week)].shape[0]
        st.write(f"**Number of complaints received last week:** {complaints_count}")

    else:
        st.write("Sorry, I can only answer questions about feedback classification or complaints count.")

