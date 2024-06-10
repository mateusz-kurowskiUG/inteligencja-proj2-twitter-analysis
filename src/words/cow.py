import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from rich import print
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main_cow():
    processed_tokens = pd.read_csv("./data/war-day/preprocessed.csv")[
        "processed_content"
    ]
    all_rows = processed_tokens.tolist()
    all_rows_splitted = [line.split(" ") for line in all_rows]
    word_list = [word for line in all_rows_splitted for word in line]
    # Generate a word cloud image
    # stop_words = [str(a) for ]

    wordcloud = WordCloud(
        max_words=100, stopwords=[*STOPWORDS, "u", "n", "s", "n't", "co", "t", "amp"]
    ).generate(" ".join(word_list))

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("./data/war-day/cow-main.png")


def israel_cow():
    # Load the CSV file
    df = pd.read_csv("./data/war-day/preprocessed.csv")

    # Load and verify the mask image
    mask_path = "./img/israel.jpg"
    mask = Image.open(mask_path)

    # Check if the image is in RGB mode
    if mask.mode != "RGB":
        print(f"Converting image mode from {mask.mode} to RGB.")
        mask = mask.convert("RGB")

    # Convert mask to numpy array
    mask = np.array(mask)

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

    wordcloud = WordCloud(
        max_words=max_words,
        stopwords=all_stopwords,
        mode="RGBA",
        mask=mask,
        background_color="white",
        colormap="Paired",
        contour_color="white",
        width=2560,
        height=1862,
    ).generate(" ".join(word_list))

    # Display the word cloud image

    fig = plt.figure(figsize=(25.60, 18.62))
    fig.suptitle(
        f"Word cloud - tweets including hash #israel ({max_words} words)", fontsize=50
    )
    image_colors = ImageColorGenerator(mask)
    plt.imshow(
        wordcloud.recolor(color_func=image_colors),
        interpolation="bilinear",
    )
    plt.axis("off")

    # Save the word cloud image
    plt.savefig("./data/war-day/cow-israel.png")


if __name__ == "__main__":
    main_cow()
    israel_cow()
