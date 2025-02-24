import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_pipeline = joblib.load("feedback_classifier.pkl")

# Load the customer segmentation data
df_segmentation = pd.read_csv("customer_segmentation.csv")

# Function to classify feedback
def classify_feedback(text):
    return model_pipeline.predict([text])[0]

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.markdown("### Example Queries:")
st.markdown("- **What category does this feedback fall under?**")
st.markdown("- **How many complaints were received last week?**")
st.markdown("- **Who are the top 5 high-value customers?**")
st.markdown("- **Which country has the most complaints?**")

# User input for feedback classification
feedback = st.text_area("Enter customer feedback:")
if st.button("Classify Feedback"):
    if feedback:
        category = classify_feedback(feedback)
        st.success(f"Feedback Category: {category}")
    else:
        st.warning("Please enter feedback text.")

# Query: How many complaints were received last week?
if st.button("Show Complaints Last Week"):
    st.write("Feature to be implemented: Fetch complaints based on date from dataset.")

# Query: Who are the top 5 high-value customers?
if st.button("Top 5 High-Value Customers"):
    df_segmentation['TotalSpent'] = df_segmentation['Quantity'] * df_segmentation['UnitPrice']
    top_customers = df_segmentation.groupby('CustomerID')['TotalSpent'].sum().sort_values(ascending=False).head(5)
    st.write(top_customers)

# Query: Which country has the most complaints?
if st.button("Country with Most Complaints"):
    st.write("Feature to be implemented: Count complaints per country from dataset.")
