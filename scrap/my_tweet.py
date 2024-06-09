from twscrape import Tweet


class MyTweet:
    def __init__(self, tweet: Tweet) -> None:
        self.date = tweet.date
        self.raw_content = tweet.rawContent
        self.retweet_count = tweet.retweetCount
        self.like_count = tweet.likeCount
        self.quote_count = tweet.quoteCount
        self.hash_tags = tweet.hashtags
        self.user_name = tweet.user.username
        self.user_description = tweet.user.rawDescription
        self.location = tweet.user.location