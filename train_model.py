import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("complaints.csv")  # Ensure this file exists

# Select relevant columns and drop missing values
df = df[['narrative', 'product']].dropna()
df.columns = ['Complaint', 'Category']  # Rename columns for clarity

# Convert text data into numerical vectors
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(df['Complaint'])
y_train = df['Category']

# Train a Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save trained model and vectorizer
joblib.dump((vectorizer, model), "bank_complaint_classifier.pkl")

print("✅ Model trained and saved successfully!")
