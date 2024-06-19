import pandas as pd
from rich import print
import matplotlib.pyplot as plt
import numpy as np
import ast


def adjust_data(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.strftime(
        "%Y-%m"
    )  # Convert to datetime type
    df = df.drop(["processed_content"], axis=1)
    df = df.drop("hash_tags", axis=1)
    df = df.sort_values(by="date")
    return df


def plot_sentiment(df: pd.DataFrame, title_part: str, file_name: str):
    df_grouped = df.groupby("date").mean()
    print(df_grouped)
    dates = df_grouped.index.astype(str)
    pos = df_grouped["pos_sentiment"]
    neu = df_grouped["neu_sentiment"]
    neg = df_grouped["neg_sentiment"]

    plt.figure(figsize=(15, 6))
    plt.plot(dates, pos, label="Positive", color="green")
    plt.plot(dates, neu, label="Neutral", color="blue")
    plt.plot(dates, neg, label="Negative", color="red")

    plt.title(f"NLTK Vader Sentiment Over Time ({title_part})")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Scores")
    plt.xticks(dates[::2], rotation=45)
    plt.axvline(
        x="2021-05", color="black", linestyle="--", label="May conflict in Gaza"
    )
    plt.axvline(
        x="2022-08", color="black", linestyle="-", label="August operation in Gaza"
    )
    plt.axvline(
        x="2023-10",
        color="blue",
        linestyle="-",
        label="October 7 Hamas massacre\n (start of the war)",
    )
    plt.axvline(
        x="2024-02",
        color="blue",
        linestyle="--",
        label="Israel opens fire to UN convoy",
    )
    plt.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()

    plt.savefig(f"./data/combined/{file_name}")


if __name__ == "__main__":
    df = pd.read_csv("./data/combined/preprocessed.csv")

    # Convert hash_tags column from string representation of list to actual list
    df["hash_tags"] = df["hash_tags"].apply(ast.literal_eval)

    new_df = adjust_data(df)
    plot_sentiment(new_df, "All tweets", "nltk-sentiment-all.png")
    print("Plot created and saved successfully!")

    # Filter for records related to Israel
    israel_only = df[
        df["hash_tags"].apply(lambda tags: any("israel" in tag.lower() for tag in tags))
    ]
    israel_df = adjust_data(israel_only)
    plot_sentiment(israel_df, "Israel-related tweets", "nltk-sentiment-israel.png")
    print("Plot created and saved successfully!")

    # Filter for records related to Palestine
    palestine_only = df[
        df["hash_tags"].apply(
            lambda tags: any("palestine" in tag.lower() for tag in tags)
        )
    ]
    palestine_df = adjust_data(palestine_only)
    plot_sentiment(
        palestine_df, "Palestine-related tweets", "nltk-sentiment-palestine.png"
    )
    print("Plot created and saved successfully!")