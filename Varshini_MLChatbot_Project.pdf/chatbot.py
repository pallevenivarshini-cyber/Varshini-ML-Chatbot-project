import json
import random
import pickle

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Load vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Generate response
def get_response(user_input):

    X = vectorizer.transform([user_input])

    tag = model.predict(X)[0]

    for intent in data['intents']:

        if intent['tag'] == tag:

            return random.choice(intent['responses'])

    return "Sorry, I don't understand."

# Chat loop
while True:

    message = input("You: ")

    if message.lower() == "quit":

        print("Bot: Goodbye!")

        break

    response = get_response(message)

    print("Bot:", response)