import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text - Tokenization
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# Function to preprocess text - Remove punctuation and special characters
def remove_punctuation(tokens):
    tokens = [token for token in tokens if token not in string.punctuation and token.isalnum()]
    return tokens

# Function to preprocess text - Convert text to lowercase
def convert_to_lowercase(tokens):
    tokens = [token.lower() for token in tokens]
    return tokens

# Function to preprocess text - Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Function to preprocess text - Stemming
def apply_stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to calculate cosine similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Streamlit App
def main():
    st.title("Text Preprocessing and Cosine Similarity")

    # Sample review text
    review_text = st.text_area("Enter review text:", "This is a sample review text. It contains some stopwords, punctuation, and special characters!")

    # Tokenization
    tokens = tokenize_text(review_text)
    st.subheader("Tokenization:")
    st.write(tokens)

    # Remove punctuation and special characters
    tokens_no_punctuation = remove_punctuation(tokens)
    st.subheader("Remove Punctuation and Special Characters:")
    st.write(tokens_no_punctuation)

    # Convert text to lowercase
    tokens_lowercase = convert_to_lowercase(tokens_no_punctuation)
    st.subheader("Convert Text to Lowercase:")
    st.write(tokens_lowercase)

    # Remove stopwords
    tokens_no_stopwords = remove_stopwords(tokens_lowercase)
    st.subheader("Remove Stopwords:")
    st.write(tokens_no_stopwords)

    # Apply stemming
    stemmed_tokens = apply_stemming(tokens_no_stopwords)
    st.subheader("Stemming:")
    st.write(stemmed_tokens)

    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    st.subheader("Preprocessed Text:")
    st.write(preprocessed_text)

    st.markdown("---")

    # Cosine Similarity Tab
    st.title("Cosine Similarity")

    # Load dataset
    data = pd.read_csv("exam.csv")  # Replace "exam.csv" with your dataset path

    # Dropdown for division names
    division_names = data['Division Name'].unique()
    selected_division = st.selectbox("Select a division name:", division_names)

    # Filter data based on selected division name
    filtered_data = data[data['Division Name'] == selected_division]

    # Display filtered data
    st.subheader("Reviews for '{}' division:".format(selected_division))
    st.write(filtered_data)

    # Remove NaN values from review text
    filtered_data = filtered_data.dropna(subset=['Review Text'])

    # Calculate cosine similarity between Review 1 and Reviews 2 to 10
    if len(filtered_data) > 1:
        st.subheader("Similarity Analysis:")
        for j in range(1, min(len(filtered_data), 10)):
            similarity_score = calculate_cosine_similarity(filtered_data.iloc[0]['Review Text'], filtered_data.iloc[j]['Review Text'])
            st.write("Cosine Similarity between Review 1 and Review {}: {:.2f}".format(j+1, similarity_score))

if __name__ == "__main__":
    main()
