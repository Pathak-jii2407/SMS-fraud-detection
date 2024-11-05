import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Ensure you have downloaded the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer with absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tfidf_path = os.path.join(current_dir, 'vectorizer.pkl')
model_path = os.path.join(current_dir, 'model.pkl')

try:
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please ensure the file is in the correct directory.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
