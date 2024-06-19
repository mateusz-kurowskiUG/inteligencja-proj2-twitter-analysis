import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from rich import print
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.words.stopwords import ALL_STOPWORDS


# Create word cloud
def create_wordcloud(
    stopwords,
    path,
    words,
    figsize,
    max_words,
    title,
    size,
    fontsize=50,
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
        max_words=max_words,
    ).generate(" ".join(words))

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{title} ({max_words} words)", fontsize=fontsize)
    plt.tight_layout(pad=0)

    if color_func is not None:
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    else:
        plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")
    plt.savefig(path)
    plt.close(fig)


# Create mask for word cloud
def create_mask(path: str):
    mask = Image.open(path)

    # Check if the image is in RGB mode
    if mask.mode != "RGB":
        print(f"Converting image mode from {mask.mode} to RGB.")
        mask = mask.convert("RGB")

    # Convert mask to numpy array
    return np.array(mask)


if __name__ == "__main__":
    # Load preprocessed data
    new_df = pd.read_csv("./data/combined/preprocessed.csv")

    # Generate word clouds
    processed_tokens = new_df["processed_content"]
    all_rows = processed_tokens.tolist()
    all_rows_splitted = [line.split(" ") for line in all_rows]
    word_list = [word for line in all_rows_splitted for word in line]

    # General word cloud
    create_wordcloud(
        ALL_STOPWORDS,
        "./data/combined/cow-main.png",
        word_list,
        (25.60, 14.40),
        100,
        "All tweets wordcloud",
        (1440, 2560),
    )

    # Word cloud for tweets containing #israel
    israel_rows = new_df[
        new_df["hash_tags"].str.contains("israel", case=False, na=False)
    ]
    israel_word_list = [
        word
        for line in israel_rows["processed_content"].tolist()
        for word in line.split(" ")
    ]
    israel_mask = create_mask("./img/israel.jpg")
    image_colors = ImageColorGenerator(israel_mask)
    create_wordcloud(
        # stopwords
        ALL_STOPWORDS,
        "./data/combined/cow-israel.png",
        israel_word_list,
        (25.60, 18.62),
        300,
        "Wordcloud - tweets including #israel tag",
        (1862, 2560),
        colormap="Paired",
        mask=israel_mask,
        color_func=image_colors,
        image_colors=image_colors,
        background_color="white",
    )

    # Word cloud for tweets containing #palestine
    palestine_rows = new_df[
        new_df["hash_tags"].str.contains("palestine", case=False, na=False)
    ]
    palestine_word_list = [
        word
        for line in palestine_rows["processed_content"].tolist()
        for word in line.split(" ")
    ]
    palestine_mask = create_mask("./img/palestine.png")
    image_colors = ImageColorGenerator(palestine_mask)
    create_wordcloud(
        # stopwords
        ALL_STOPWORDS,
        "./data/combined/cow-palestine.png",
        palestine_word_list,
        (25.60, 12.80),
        200,
        "Wordcloud - tweets including #palestine tag",
        (1280, 2560),
        colormap="Paired",
        mask=palestine_mask,
        color_func=image_colors,
        image_colors=image_colors,
        background_color="white",
    )
