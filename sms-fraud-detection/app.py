import streamlit as st
import pickle
import nltk
import string
# import psycopg2
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources if not already downloaded
nltk.download('punkt')     # This is essential for tokenization
nltk.download('stopwords')  # Download stopwords if not already downloaded
nltk.download('punkt_tab')

# Connect to the PostgreSQL database
# hostname = 'dpg-csksslu8ii6s7380o05g-a.oregon-postgres.render.com'
# port = 5432
# database = 'fraud_detection_binary_files'
# username = 'fraud_detection_binary_files_user'
# password = 'p3Qtz54g2Btije8jb4BCjuEKJWSoGyTA'

# connection = psycopg2.connect(host=hostname, port=port, database=database, user=username, password=password)
# cursor = connection.cursor()

st.title("Email/SMS Spam Classifier")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
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

# def retrieve_model(model_name):
#     cursor.execute("SELECT model_data FROM model_storage WHERE model_name = %s", (model_name,))
#     model_data = cursor.fetchone()
#     return model_data[0] if model_data else None

# def load_model_from_db(model_name):
#     model_data = retrieve_model(model_name)
#     if model_data:
#         return pickle.loads(model_data)
#     else:
#         print(f"{model_name} not found in the database.")
#         return None

# Load models from the database
# tfidf = load_model_from_db('vectorizer')  # Use the model name
# model = load_model_from_db('model')         # Use the model name

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    if tfidf and model:  # Ensure models are loaded
        transformed_sms = transform_text(input_sms)
        vector_inp = tfidf.transform([transformed_sms])
        
        result = model.predict(vector_inp)[0]

        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
    else:
        st.error("Model loading failed. Unable to make predictions.")

# Clean up
# cursor.close()
# connection.close()
