import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rich import print
from autocorrect import Speller
import re
import numpy as np
from src.words.stopwords import ALL_STOPWORDS

# Initialize necessary components
analyzer = SentimentIntensityAnalyzer()
speller = Speller(fast=True)
lemmatizer = WordNetLemmatizer()

# Custom stopwords list


# Combine datasets
def combine() -> pd.DataFrame:
    after_war = pd.read_csv("./data/war-day/dataset.csv")
    before_war = pd.read_csv("./data/before-war/dataset.csv")
    combined_df = pd.concat([before_war, after_war]).drop("Unnamed: 0", axis=1)
    combined_df = combined_df.reset_index(drop=True)
    combined_df.to_csv("./data/combined/dataset.csv", index=False)
    return combined_df


# Preprocessing function
def preprocess_text(text: str):
    if pd.isnull(text):  # Check for NaN values
        return ""
    text = text.encode("ascii", errors="ignore").decode()
    # Remove URLs
    text = re.sub(r"https?:\/\/\S+|www\.\S+", " ", text)

    # Spelling correction
    text = speller(text.lower().strip())
    # Tokenize
    tokens = word_tokenize(text)

    # Stopwords and punctuation removal
    filtered_tokens = [token for token in tokens if token not in ALL_STOPWORDS]

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # print(lemmatized_tokens)
    return " ".join(lemmatized_tokens)


# Sentiment analysis function
def get_sentiment(text):
    if pd.isnull(text) or text.strip() == "":
        return np.nan
    scores = analyzer.polarity_scores(text)
    return 1 if scores["pos"] > 0 else 0


if __name__ == "__main__":
    df = combine()

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
    new_df.to_csv("./data/combined/preprocessed.csv", index=False)
    print("Preprocessing done!")
