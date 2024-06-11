import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rich import print
import pandas as pd
import string
from autocorrect import Speller
import re
import numpy as np

analyzer = SentimentIntensityAnalyzer()
speller = Speller(fast=True)


# Preprocessing function
def preprocess_text(text: str):
    if pd.isnull(text):  # Check for NaN values
        return ""

    # Remove URLs
    text = re.sub(r"""https?:\/\/\S+|www\.\S+""", " ", text)

    # Spelling correction
    text = speller(text.lower().strip())

    # Tokenize
    tokens = word_tokenize(text)

    # Stopwords and punctuation removal
    new_stopwords = stopwords.words("english") + list(string.punctuation)
    new_stopwords.extend(
        [
            "'",
            "`",
            "...",
            ".",
            '"',
            "``",
            "..",
            "”",
            "''",
            "“",
            "’",
            ",",
            "co",
            "http",
            "amp",
            "u",
            "n",
            "s",
            "n't",
            "t",
            " ",
            "\n",
        ]
    )
    filtered_tokens = [token for token in tokens if token not in new_stopwords]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return " ".join(lemmatized_tokens)


# Sentiment analysis function
def get_sentiment(text):
    if pd.isnull(text) or text.strip() == "":
        return np.nan
    scores = analyzer.polarity_scores(text)
    return 1 if scores["pos"] > 0 else 0


# Load dataset
def load_dataset():
    return pd.read_csv("./data/war-day/df.csv")

if __name__ == "__main__":
    df = load_dataset()

    # Create new DataFrame
    new_df = pd.DataFrame()
    new_df["date"] = df["date"]

    # Process content
    new_df["processed_content"] = df["raw_content"].apply(preprocess_text)

    # Drop rows with NaN or empty processed_content
    new_df = new_df[new_df["processed_content"].str.strip() != ""]

    new_df["hash_tags"] = df["hash_tags"]

    # Sentiment analysis
    new_df["content_sentiment"] = new_df["processed_content"].apply(get_sentiment)

    # Drop rows with NaN sentiment
    new_df = new_df.dropna(subset=["content_sentiment"])

    # Save to CSV
    new_df.to_csv("./data/war-day/preprocessed.csv", index=False)
    print("Preprocessing done!")
