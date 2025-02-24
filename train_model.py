import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("customer_feedback.csv")

# Preprocess function
def preprocess_text(text):
    import re
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        return text
    return ""

# Apply preprocessing
df['Processed_Reviews'] = df['Reviews'].apply(preprocess_text)

# Define categories based on keywords
def categorize_feedback(text):
    if any(word in text for word in ["bad", "poor", "slow", "broken", "issue", "problem", "worst"]):
        return "Complaint"
    elif any(word in text for word in ["should", "could", "better", "improve", "suggest", "recommend"]):
        return "Suggestion"
    elif any(word in text for word in ["good", "great", "excellent", "amazing", "best", "love", "happy"]):
        return "Praise"
    return "Neutral"

# Apply categorization
df["Category"] = df["Processed_Reviews"].apply(categorize_feedback)
df_filtered = df[df["Category"] != "Neutral"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_filtered["Processed_Reviews"], df_filtered["Category"], test_size=0.2, random_state=42)

# Define and train Naive Bayes model
model_pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, "feedback_classifier.pkl")

print("Model training complete. Accuracy:", model_pipeline.score(X_test, y_test))
