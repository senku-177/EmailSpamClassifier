import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import os
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)

# Other imports and code for your application


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalpha():
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

tfidf = pickle.load(open('Vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_subject=st.text_input("Enter the Subject")
input_email = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_email = transform_text(input_email)
    transform_subject=transform_text(input_subject)
    print(transform_subject+transformed_email)
    # 2. vectorize
    vector_input = tfidf.transform([transform_subject+transformed_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
