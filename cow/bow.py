import pandas as pd
from rich import print


def load_preprocessed():
    return pd.read_csv("./data/war-day/preprocessed.csv")


if __name__ == "__main__":
    df = load_preprocessed()
    print(df)