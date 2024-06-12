from rich import print
import pandas as pd


def read_bag_of_words() -> pd.DataFrame:
    return pd.read_csv("./data/war-day/bow.csv")


if __name__ == "__main__":
    df = read_bag_of_words()
    print(df)