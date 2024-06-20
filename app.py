import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load the trained model and TF-IDF vocabulary
model = pickle.load(open("model_sentimen.sav", "rb"))
vocab = pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))
loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=set(vocab))


factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords_ind = stopwords.words('indonesian')

# Define text preprocessing using imported functions
def text_preprocessing_process(text):
    text = casefolding(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

def casefolding(text):
  text = text.lower() #mengubah kalimat menjadi huruf kecil
  text = re.sub(r'https?://\S+|www\.\S+', '', text) #menghapus urll dari kalimat
  text = re.sub(r'[-+]?[0-9]+', '', text) #mmenghapus angka dari kalimat
  text = re.sub(r'[^\w\s]', '', text) #menghapus tanda baca dari kalimat
  text = text.strip()
  return text

def remove_stop_words(text):
  clean_words = []
  text = text .split()
  for word in text:
    if word not in stopwords_ind:
      clean_words.append(word)
  return " ".join(clean_words)


def stemming(text):
  text = stemmer.stem(text)
  return text

# Define Streamlit app
st.title("Sentiment Analysis Hotel Review")

text_input = st.text_area("Enter the text to analyze")
 
if st.button("Predict"):
    processed_text = text_preprocessing_process(text_input)

    loaded_vec.fit([processed_text])
    transformed_text = loaded_vec.transform([processed_text])
    result = model.predict(loaded_vec.fit_transform([processed_text]))
    
    if result == 'positif':
        s = "Sentimen Positif"
    elif result == 'negatif':
        s = "Sentimen Negatif"
    else:
        s = "Sentimen Netral"
    
    st.write(f"Hasil Prediksi: {s}")
