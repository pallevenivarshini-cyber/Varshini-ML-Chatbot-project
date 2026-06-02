import streamlit as st
import json
import random
import pickle

# Page settings
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Function to get chatbot response
def get_response(user_input):

    X = vectorizer.transform([user_input])

    tag = model.predict(X)[0]

    for intent in data['intents']:

        if intent['tag'] == tag:

            return random.choice(intent['responses'])

    return "Sorry, I don't understand."

# Sidebar
st.sidebar.title("About")
st.sidebar.write("ML-Based AI Chatbot")
st.sidebar.write("Built using Python and Streamlit")

# Main title
st.title("🤖 AI Chatbot")

# Chat history
if "messages" not in st.session_state:

    st.session_state.messages = []

# Input box
user_input = st.text_input("Type your message")

# Send button
if st.button("Send"):

    if user_input != "":

        response = get_response(user_input)

        st.session_state.messages.append(("You", user_input))

        st.session_state.messages.append(("Bot", response))

# Display chat messages
for sender, message in st.session_state.messages:

    if sender == "You":

        st.markdown(
            f"""
            <div style="
                background-color:#d1e7dd;
                color:black;
                padding:10px;
                border-radius:10px;
                margin:10px 0;
            ">
            <b>You:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="
                background-color:#f8d7da;
                color:black;
                padding:10px;
                border-radius:10px;
                margin:10px 0;
            ">
            <b>Bot:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )