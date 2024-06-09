import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
import pandas as pd
import json
from dotenv import load_dotenv
import os
from rich import print


def load_settings() -> dict[str]:
    with open("./data/settings.json") as settings:
        return json.loads(settings.read())


async def main():
    load_dotenv()
    TWITTER_USER = os.getenv("TWITTER_USER")
    TWITTER_PASS = os.getenv("TWITTER_PASS")
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")

    api = API()  # or API("path-to.db") - default is `accounts.db`

    # ADD ACCOUNTS (for CLI usage see BELOW)
    await api.pool.add_account(TWITTER_USER, TWITTER_PASS, EMAIL_USER, EMAIL_PASS)
    await api.pool.login_all()


if __name__ == "__main__":
    settings = load_settings()
    # asyncio.run(main())
    print(settings)
