import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt","rb"))
    return nb_model

def toxicity_detection(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    model = load_model()
    #predict the class of the input 
    prediction = model.predict(text_tfidf)

    #map the predicted class to a string
    class_name = "Toxic/Hate" if prediction == 1 else "Not a Toxic or hate comment"

    return class_name

st.header("Hate/Toxic Comment Detection")


st.subheader("Mini Project ")


text_input = st.text_input("Enter your Text : ")

if text_input is not None:
    if st.button("Analyse"):
        result  = toxicity_detection(text_input)
        st.subheader("Result : ")
        st.info("This comment is " + result + ".")

