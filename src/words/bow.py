import pandas as pd
from rich import print
from sklearn.feature_extraction.text import CountVectorizer


# Function to load preprocessed data
def load_preprocessed():
    return pd.read_csv("./data/war-day/preprocessed.csv")

if __name__ == "__main__":
    # Load the data
    df = load_preprocessed()

    # Check if the 'processed_content' column exists and is not empty
    if "processed_content" not in df.columns:
        print(
            "[bold red]Error: 'processed_content' column not found in the CSV file.[/bold red]"
        )
    elif df["processed_content"].isnull().all():
        print(
            "[bold red]Error: 'processed_content' column is empty or all values are null.[/bold red]"
        )
    else:
        # Print the first few rows of the DataFrame for debugging
        print("[bold green]First few rows of the input data:[/bold green]")
        print(df.head())

        # Initialize the CountVectorizer
        count_vect = CountVectorizer()

        # Fit and transform the processed content column
        count_matrix = count_vect.fit_transform(df["processed_content"])

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

            # Print the resulting DataFrame
            print("[bold green]CountVectorizer result:[/bold green]")
            df_count.to_csv("./data/war-day/bow.csv")
