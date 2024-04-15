import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

dataset_path = Path("data/preprocessed_dataset.csv")
gender_table_path = Path("data/gender_table.csv")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def match_firstname(firstname: str, table: pd.DataFrame) -> str:
    # Find the corresponding row in the table with the firstname via fuzzy matching
    choices = table["firstname"].tolist()
    match = process.extractOne(firstname, choices, scorer=fuzz.ratio)
    if match is None:
        return None
    else:
        selected_row = table[table["firstname"].str.contains(match[0])]
        return selected_row["sex"].values[0]


def predict(df: pd.DataFrame, table: pd.DataFrame) -> None:
    for i, row in df.iterrows():
        # Get the firstname
        firstname = row["firstname"]

        # Find the corresponding row in the table with the firstname via fuzzy matching
        sex = match_firstname(firstname, table)
        print(firstname, sex)


def main():
    # Load the preprocessed data
    df = load_data(dataset_path)
    gender_table = load_data(gender_table_path)

    _ = predict(df, gender_table)


if __name__ == "__main__":
    main()
