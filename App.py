import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)


# Load trained model and TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)


# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Title
st.title("📰 Fake News Detection")

st.write(
    "Enter a news article below to predict whether it is Real or Fake."
)

# Text input
article = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# Prediction button
if st.button("Predict"):

    if article.strip() == "":
        st.warning("Please enter a news article.")

    else:
        # Clean text
        cleaned_article = clean_text(article)

        # Convert text into TF-IDF features
        article_tfidf = tfidf.transform([cleaned_article])

        # Make prediction
        prediction = svm_model.predict(article_tfidf)

        # Display result
        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("⚠️ Fake News")
