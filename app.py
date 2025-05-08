import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Prediction function
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    return result

# UI Setup
st.set_page_config(page_title="Spam SMS Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Spam SMS Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# About section
with st.expander(" About Spam Detector"):
    st.write("""
        This app uses a machine learning model to analyze SMS messages and classify them as **Spam** or **Not Spam**.
        It helps users avoid phishing, fraud, and promotional spam by scanning message content intelligently.
    """)

# Input box
st.markdown("#### Enter your SMS message:")
input_sms = st.text_area("")

# Detect button
if st.button(" Detect Spam"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_message(input_sms)
        if result == 1:
            st.error("**Danger! This message is likely SPAM.**\n\nPlease avoid clicking suspicious links or sharing personal info.")
        else:
            st.success("**Safe! This message is NOT spam.**\n\nNo immediate threats detected.")

# Footer
st.markdown("---")

