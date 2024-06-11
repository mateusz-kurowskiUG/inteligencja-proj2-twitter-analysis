import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from rich import print
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Iterable


def create_wordcloud(
    stopwords: Iterable,
    path: str,
    words: list[str],
    figsize: tuple[int, int],
    max_words: int,
    title: str,
    size: tuple[float, float],
    fontsize: int = 50,
    colormap=None,
    background_color="black",
    mask=None,
    mode="RGBA",
    color_func=None,
    image_colors=None,
):
    wordcloud = WordCloud(
        stopwords=stopwords,
        height=size[0],
        width=size[1],
        mask=mask,
        background_color=background_color,
        colormap=colormap,
        mode=mode,
        color_func=color_func,
    ).generate(" ".join(words))
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"{title} ({max_words} words)",
        fontsize=fontsize,
    )
    plt.tight_layout(pad=0)
    if color_func is not None:
        plt.imshow(
            wordcloud.recolor(color_func=image_colors),
            interpolation="bilinear",
        )
    else:
        plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")
    plt.savefig(path)


def main_cow():
    processed_tokens = pd.read_csv("./data/war-day/preprocessed.csv")[
        "processed_content"
    ]
    all_rows = processed_tokens.tolist()
    for line in all_rows:
        if isinstance(line, float):
            print(line)
    all_rows_splitted = [line.split(" ") for line in all_rows]
    word_list = [word for line in all_rows_splitted for word in line]
    # Generate a word cloud image
    # stop_words = [str(a) for ]
    max_words = 200
    stopwords = [*STOPWORDS, "u", "n", "s", "n't", "co", "t", "amp"]
    path = "./data/war-day/cow-main.png"
    figsize = (25.60, 14.40)
    width = 2560
    height = 1440
    title = "All tweets wordcloud"
    create_wordcloud(
        stopwords, path, word_list, figsize, max_words, title, (height, width)
    )


def israel_cow():
    # Load the CSV file
    df = pd.read_csv("./data/war-day/preprocessed.csv")
    # Filter rows where hashtags contain "israel"
    israel_rows = df[df["hash_tags"].str.contains("israel", case=False, na=False)]
    # Extract processed tokens from the filtered rows
    processed_tokens = israel_rows["processed_content"]
    # Convert processed tokens to a list of words
    all_rows = processed_tokens.tolist()
    all_rows_splitted = [line.split(" ") for line in all_rows]
    word_list = [word for line in all_rows_splitted for word in line]

    # Generate a word cloud image
    stopwords = set(STOPWORDS)
    custom_stopwords = set(["u", "n", "s", "n't", "co", "t", "amp"])
    all_stopwords = stopwords.union(custom_stopwords)

    max_words = 500
    mask = create_mask("./img/israel.jpg")

    max_words = max_words
    stopwords = all_stopwords
    mode = "RGBA"
    background_color = "white"
    colormap = "Paired"
    size = (2560, 1862)
    title = "Wordcloud - tweets including #israel tag"
    # Display the word cloud image
    path = "./data/war-day/cow-israel.png"
    figsize = (25.60, 18.62)
    image_colors = ImageColorGenerator(mask)

    create_wordcloud(
        all_stopwords,
        path,
        word_list,
        figsize,
        max_words,
        title,
        size,
        image_colors=image_colors,
        mask=mask,
        color_func=image_colors,
        mode=mode,
        background_color=background_color,
        colormap=colormap,
    )


def palestine_cow():
    # Load the CSV file
    df = pd.read_csv("./data/war-day/preprocessed.csv")
    # Filter rows where hashtags contain "israel"
    palestine = df[df["hash_tags"].str.contains("palestine", case=False, na=False)]
    # Extract processed tokens from the filtered rows
    processed_tokens = palestine["processed_content"]
    # Convert processed tokens to a list of words
    all_rows = processed_tokens.tolist()
    all_rows_splitted = [line.split(" ") for line in all_rows]
    word_list = [word for line in all_rows_splitted for word in line]

    # Generate a word cloud image
    stopwords = set(STOPWORDS)
    custom_stopwords = set(["u", "n", "s", "n't", "co", "t", "amp"])
    all_stopwords = stopwords.union(custom_stopwords)

    max_words = 500
    mask = create_mask("./img/palestine.png")

    max_words = max_words
    stopwords = all_stopwords
    mode = "RGBA"
    background_color = "white"
    colormap = "Paired"
    size = (2560, 1280)
    title = "Wordcloud - tweets including #palestine tag"
    # Display the word cloud image
    path = "./data/war-day/cow-palestine.png"
    figsize = (25.60, 12.80)
    image_colors = ImageColorGenerator(mask)

    create_wordcloud(
        all_stopwords,
        path,
        word_list,
        figsize,
        max_words,
        title,
        size,
        image_colors=image_colors,
        mask=mask,
        color_func=image_colors,
        mode=mode,
        background_color=background_color,
        colormap=colormap,
    )


def create_mask(path: str):
    mask = Image.open(path)

    # Check if the image is in RGB mode
    if mask.mode != "RGB":
        print(f"Converting image mode from {mask.mode} to RGB.")
        mask = mask.convert("RGB")

    # Convert mask to numpy array
    return np.array(mask)


if __name__ == "__main__":
    main_cow()
    israel_cow()
    palestine_cow()
