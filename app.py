import streamlit as st
import joblib
import re
import string

model = joblib.load('trained_model.pkl')
vec = joblib.load('tfidf_vectorizer.pkl')

# preprocessing the userinput
def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System", 
    page_icon="📰", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a page:", ["Project Overview", "Fake News Classifier"])

# Add custom CSS styles for better visuals
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .title {
            color: #4A90E2;
            text-align: center;
            font-weight: bold;
            font-size: 40px;
        }
        .subtitle {
            color: #6D6D6D;
            text-align: center;
            font-size: 20px;
        }
        .content {
            font-size: 18px;
            line-height: 1.6;
        }
        .sidebar .sidebar-content {
            background-color: #f2f2f2;
        }
    </style>
""", unsafe_allow_html=True)

# Page 1: Project Overview
if page == "Project Overview":
    st.markdown('<div class="title">Fake News Detection System By Unakalamba Onyekachi Cosmas</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">📰 Powered by Natural Language Processing & Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content">
        Welcome to the **Fake News Detection System**! This application helps combat misinformation by identifying whether a piece of news is fake or authentic.

        ### Objectives 🎯
        - Detect fake news using advanced AI techniques.
        - Provide users with quick and accurate results.
        - Raise awareness about the dangers of misinformation.

        ### How it Works 🛠️
        1. Input a news headline or article.
        2. The system processes the text using a machine learning model.
        3. The result indicates whether the news is **Fake** or **Not Fake**, along with a confidence score.
        </div>
        """,
        unsafe_allow_html=True
    )

# Page 3: Fake News Classifier
elif page == "Fake News Classifier":
    st.markdown('<div class="title">Fake News Classifier</div>', unsafe_allow_html=True)
    st.write("🌟 Enter a news headline or article below, and the system will classify it as **Fake** or **Not Fake**.")

    # Input box for news article
    user_input = st.text_area("🖊️ Enter News Headline or Article:", height=200, placeholder="Type or paste your news text here...")

    # Button and results section
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Classify"):
            if user_input.strip():
                # Preprocess the text
                cleaned_text = preprocess(user_input)
                
                # Vectorize the cleaned text using the loaded vectorizer
                vec_text = vec.transform([cleaned_text])  # Transforming the text
                
                # Get the model's prediction
                prediction = model.predict(vec_text)[0]  # Predicting and taking the first result
                
                # Display the result
                if prediction == 0:
                    st.markdown('<p style="font-size:50px;color:red;font-weight:bold;">The news is classified as: Fake</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:50px;color:green;font-weight:bold;">The news is classified as: Legit</p>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a valid news headline or article.")
    
