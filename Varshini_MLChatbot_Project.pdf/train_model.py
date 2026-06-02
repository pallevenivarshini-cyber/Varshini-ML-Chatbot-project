import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open('intents.json') as file:
    data = json.load(file)

patterns = []
tags = []

# Prepare training data
for intent in data['intents']:

    for pattern in intent['patterns']:

        patterns.append(pattern)

        tags.append(intent['tag'])

# Convert text to vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(patterns)

# Train ML model
model = LogisticRegression()

model.fit(X, tags)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Save vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained successfully!")