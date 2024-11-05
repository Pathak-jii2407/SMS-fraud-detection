import streamlit as st
import pickle
import nltk
import string
import psycopg2
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Database connection details
hostname = 'dpg-csksslu8ii6s7380o05g-a.oregon-postgres.render.com'
port = 5432
database = 'fraud_detection_binary_files'
username = 'fraud_detection_binary_files_user'
password = 'p3Qtz54g2Btije8jb4BCjuEKJWSoGyTA'

connection = psycopg2.connect(host=hostname, port=port, database=database, user=username, password=password)
cursor = connection.cursor()

# nltk.download('stopwords')
# nltk.download('punkt')

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

def retrieve_model(model_name):
    cursor.execute("SELECT model_data FROM model_storage WHERE model_name = %s", (model_name,))
    model_data = cursor.fetchone()
    return model_data[0] if model_data else None

def load_model_from_db(model_name):
    model_data = retrieve_model(model_name)
    if model_data:
        return pickle.loads(model_data)  
    else:
        print(f"{model_name} not found in the database.")
        return None

# Load models from the database
tfidf = load_model_from_db(r'C:\Users\saura\OneDrive\Documents\GitHub\SMS-fraud-detection\sms-fraud-detection\vectorizer.pkl')
model = load_model_from_db(r'C:\Users\saura\OneDrive\Documents\GitHub\SMS-fraud-detection\sms-fraud-detection\model.pkl')

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_inp = tfidf.transform([transformed_sms])

    result = model.predict(vector_inp)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')

cursor.close()
connection.close()
