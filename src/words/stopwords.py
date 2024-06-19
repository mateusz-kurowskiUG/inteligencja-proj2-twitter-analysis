from stopwordsiso import stopwords as iso_stopwords
from wordcloud import STOPWORDS
import string
from nltk.corpus import stopwords

# List of languages for which to get stopwords
languages = ["pl", "en", "de", "ar", "ru", "fr"]

# Create a flattened list of stopwords for the specified languages
MY_STOPWORDS = [word for country in languages for word in iso_stopwords(country)]
ALL_STOPWORDS = set(
    [
        *MY_STOPWORDS,
        *STOPWORDS,
        *stopwords.words("english"),
        *list(string.punctuation),
        "al",
        "amp",
        "n't",
        "s",
        "'s",
        "``",
        "..",
        "...",
        "''",
    ]
)