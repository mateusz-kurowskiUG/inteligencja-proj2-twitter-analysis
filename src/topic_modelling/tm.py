from rich import print
import pandas as pd
import spacy
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import PIL.Image as Img
import io


def read_bag_of_words() -> pd.DataFrame:
    return pd.read_csv("./data/combined/bow.csv")


def read_preprocessed() -> pd.DataFrame:
    return pd.read_csv("./data/combined/preprocessed.csv")


def save_bytes_as_img(img: bytes, path: str):
    image = Img.open(io.BytesIO(img))
    image.save(path)


def save_html(html: str, path: str):
    with open(path, "w") as file:
        file.write(html)


def create_tm():
    bow = read_bag_of_words()
    df = read_preprocessed()

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    topic_model = BERTopic(language="multilingual")
    topics, probs = topic_model.fit_transform(df["processed_content"])
    fig = topic_model.visualize_topics()
    nlp.to_disk("./data/combined/tm.model")
    fig.show()
    # Uncomment the following lines if you want to save the image and HTML
    # img = fig.to_image()
    # save_bytes_as_img(img, "./data/combined/tm_plot.png")

    html = fig.to_html()
    save_html(html, "./data/combined/tm.html")

    # topic_info = topic_model.get_topic_info()
    # freq = topic_model.get_topic_freq()
    # # tree = topic_model.get_topic_tree()
    # print(freq)
    # # print(tree)


def read_tm():
    # Deserialize later, e.g. in a new process
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(bytes_data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    # https://spacy.io/usage/saving-loading


if __name__ == "__main__":
    fig = read_tm()
    fig.show()
