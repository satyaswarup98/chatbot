import pandas as pd
import joblib
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset
df = pd.read_csv("customer_feedback.csv")

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text using NLTK
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        words = word_tokenize(text)  # Tokenize text
        words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove stopwords
        return ' '.join(words)
    return ""

# Apply preprocessing
df['Processed_Reviews'] = df['Reviews'].apply(preprocess_text)

# Define categories using NLTK
def categorize_feedback(text):
    if any(word in text for word in ["bad", "poor", "slow", "broken", "issue", "problem", "worst", "terrible"]):
        return "Complaint"
    elif any(word in text for word in ["should", "could", "better", "improve", "suggest", "recommend"]):
        return "Suggestion"
    elif any(word in text for word in ["good", "great", "excellent", "amazing", "best", "love", "happy", "awesome"]):
        return "Praise"
    return "Neutral"

# Apply categorization
df["Category"] = df["Processed_Reviews"].apply(categorize_feedback)
df_filtered = df[df["Category"] != "Neutral"]  # Remove neutral reviews

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_filtered["Processed_Reviews"], df_filtered["Category"], test_size=0.2, random_state=42)

# Define and train Naive Bayes model with TF-IDF
model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, "feedback_classifier.pkl")

print("Model training complete. Accuracy:", model_pipeline.score(X_test, y_test))
