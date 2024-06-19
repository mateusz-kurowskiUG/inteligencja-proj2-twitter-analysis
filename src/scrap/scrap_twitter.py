import asyncio
from twscrape import API, gather, Tweet
import pandas as pd
import json
from dotenv import load_dotenv
import os
from rich import print
from src.scrap.my_tweet import MyTweet


def load_settings() -> dict[str]:
    with open("./data/settings.json") as settings:
        return json.loads(settings.read())


async def load_tweets_by_tag(api: API, tag: str, limit: int):
    return await gather(api.search(f"{tag} lang:en until:2023-10-07", limit=limit))


def save_to_dataset(results: list[dict[str, any]], file_name: str):
    pd.DataFrame(results).to_csv(f"{file_name}.csv")


async def login_accounts(api: API):
    load_dotenv()
    for i in range(1, 10):
        TWITTER_USER = os.getenv(f"TWITTER_USER{i}")
        TWITTER_PASS = os.getenv(f"TWITTER_PASS{i}")
        EMAIL_USER = os.getenv(f"EMAIL_USER{i}")
        EMAIL_PASS = os.getenv(f"EMAIL_PASS{i}")
        await api.pool.add_account(TWITTER_USER, TWITTER_PASS, EMAIL_USER, EMAIL_PASS)
    await api.pool.login_all()


async def main(settings: dict[str]):
    api = API()  # or API("path-to.db") - default is `accounts.db`

    await login_accounts(api)

    tags, limit = settings["tags"], settings["limit"]
    results: list[dict[str, any]] = []
    dataset_info = []
    for tag in tags:
        print(f"Started loading tag: {tag}")
        tweets = await load_tweets_by_tag(api, tag, limit)
        print(f"Completed loading tag: {tag}")
        cleared_tweets = [MyTweet(tweet).__dict__ for tweet in tweets]
        results.extend(cleared_tweets)
        dataset_info.append({"tag": tag, "count": len(cleared_tweets)})
        print(f"Added tag: {tag} to dataset")
    save_to_dataset(results, "./data/dataset")
    save_to_dataset(dataset_info, "./data/dataset_info")


if __name__ == "__main__":
    settings = load_settings()
    asyncio.run(main(settings))
