import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = pickle.load(open(r"C:\Users\Lenovo\OneDrive\Documents\penambanganteks\model_sentimen.sav", "rb"))
# Load the TF-IDF vocabulary
vocab = pickle.load(open(r"C:\Users\Lenovo\OneDrive\Documents\penambanganteks\new_selected_feature_tf-idf.sav", "rb"))
loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=vocab)

def text_preprocessing_process(text):
    # Define your preprocessing steps here
    # Assuming casefolding, remove_stop_words, and stemming functions are defined
    text = casefolding(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

# Define Streamlit app
st.title("Sentiment Analysis App")

text_input = st.text_area("Enter the text to analyze")

if st.button("Predict"):
    if text_input:
        processed_text = text_preprocessing_process(text_input)
        transformed_text = loaded_vec.transform([processed_text])
        result = model.predict(transformed_text)
        
        if result == 'positif':
            s = "Sentimen Positif"
        elif result == 'negatif':
            s = "Sentimen Negatif"
        else:
            s = "Sentimen Netral"
        
        st.write(f"Hasil Prediksi: {s}")
    else:
        st.write("Please enter text to analyze")

