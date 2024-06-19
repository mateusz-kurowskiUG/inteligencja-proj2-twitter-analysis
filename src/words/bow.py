import pandas as pd
from rich import print
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib as mpl


# Function to load preprocessed data
def load_preprocessed():
    return pd.read_csv("./data/combined/preprocessed.csv")


def sum_sort_key(e):
    return e[1]


if __name__ == "__main__":
    # Load the data
    df = load_preprocessed()
    # Check if the 'processed_content' column exists and is not empty
    if "processed_content" not in df.columns:
        print("Error: 'processed_content' column not found in the CSV file.")
    elif df["processed_content"].isnull().all():
        print("Error: 'processed_content' column is empty or all values are null.")
    else:
        # Print the first few rows of the DataFrame for debugging
        print("First few rows of the input data:")
        # Initialize the CountVectorizer
        count_vect = CountVectorizer()
        # Fit and transform the processed content column
        count_matrix = count_vect.fit_transform(
            df["processed_content"].values.astype("U")
        )

        # Convert the count matrix to a dense array
        count_array = count_matrix.toarray()
        # Check if the count array is empty
        if count_array.size == 0:
            print(
                "[bold red]Error: The count array is empty after transformation. Check the input data and CountVectorizer settings.[/bold red]"
            )
        else:
            # Create a DataFrame from the dense array with appropriate column names
            df_count = pd.DataFrame(
                data=count_array, columns=count_vect.get_feature_names_out()
            )
            summed = count_array.sum(axis=0)
            summed_with_labels = list(zip(count_vect.get_feature_names_out(), summed))
            summed_with_labels.sort(key=sum_sort_key, reverse=True)
            head_20 = summed_with_labels[:20]
            head_20_X = [x for x, _ in head_20]
            head_20_y = [y for _, y in head_20]
            print(head_20_X)
            plt.figure(figsize=(19, 10))
            plt.bar(head_20_X, head_20_y)
            plt.title("Most common words in all gathered tweets")
            plt.xlabel("Word")
            plt.ylabel("Occurencies")

            plt.autoscale(tight=True)
            plt.savefig("./data/combined/most-common-words.png")
            # Print the resulting DataFrame
            print("[bold green]CountVectorizer result:[/bold green]")
            df_count.to_csv("./data/combined/bow.csv")
