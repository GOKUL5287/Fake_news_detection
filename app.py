import pickle
import re
from pathlib import Path

import streamlit as st


# Configure the browser tab before creating other Streamlit elements.
st.set_page_config(page_title="Fake News Detection 📰", page_icon="📰")

st.title("Fake News Detection")
st.write(
    "Paste a news article below to predict whether it is Real News or Fake News."
)


def clean_text(text):
    """Apply the same basic text cleaning used before TF-IDF vectorization."""
    text = text.lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)  # Remove URLs.
    text = re.sub(r"<.*?>", " ", text)  # Remove HTML tags.
    text = re.sub(r"[^a-z\s]", " ", text)  # Keep letters and whitespace.
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace.
    return text


# Load the saved model files from the same folder as this application.
project_folder = Path(__file__).resolve().parent
model_path = project_folder / "svm_model.pkl"
vectorizer_path = project_folder / "tfidf_vectorizer.pkl"

try:
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    with vectorizer_path.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as error:
    st.error(
        "Could not load the trained model files. Make sure "
        "`svm_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder "
        "as `app.py`."
    )
    st.caption(f"Details: {error}")
    st.stop()


article = st.text_area("News article", height=300)

if st.button("Predict"):
    if not article.strip():
        st.warning("Please enter a news article before predicting.")
    else:
        cleaned_article = clean_text(article)
        article_vector = vectorizer.transform([cleaned_article])
        prediction = model.predict(article_vector)[0]

        if prediction == 1:
            st.success("Real News")
        elif prediction == 0:
            st.error("Fake News")
        else:
            st.warning("The model returned an unexpected prediction.")
