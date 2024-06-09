import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rich import print
import pandas as pd
import string
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
import text2emotion as te

analyzer = SentimentIntensityAnalyzer()
speller = Speller(fast=True)
# Use a pipeline as a high-level helper


def preprocess_text(text: str):
    # remove spells
    spells_removed = speller(text.lower().strip())
    # tokenize
    tokens = word_tokenize(spells_removed)
    # remove stop words

    new_stopwords = stopwords.words("english") + list(string.punctuation)
    new_stopwords.extend(
        ["'", "`", "...", ".", '"', "``", "..", "”", "''", "“", "’", ","]
    )
    filtered_tokens = [token for token in tokens if token not in new_stopwords]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    return lemmatized_tokens


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores["pos"] > 0 else 0
    return sentiment


def load_dataset():
    return pd.read_csv("./data/war-day/df.csv")


if __name__ == "__main__":
    # download if not present on the device
    # nltk.download()
    df = load_dataset()

    new_df = pd.DataFrame()
    new_df["date"] = df["date"]
    new_df["content_tokens"] = df["raw_content"].apply(preprocess_text)
    new_df["processed_content"] = new_df["content_tokens"].apply(" ".join)
    # new_df["user_desc_preprocessed"] = (
    #     df["user_description"].apply(str).apply(preprocess_text)
    # )
    new_df["hash_tags"] = df["hash_tags"]
    new_df["content_sentiment"] = new_df["processed_content"].apply(get_sentiment)
    # new_df["user_desc_sentiment"] = new_df["processed_user_desc"].apply(
    #     get_sentiment
    # )
    # vectorizer = CountVectorizer()
    # x = vectorizer.fit_transform(new_df["processed_content"])
    # print(x)
    new_df.to_csv("./data/war-day/preprocessed.csv")