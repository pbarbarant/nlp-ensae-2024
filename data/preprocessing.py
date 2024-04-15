import pandas as pd
from pathlib import Path


def load_data(data_path: Path, sep: str = ",") -> pd.DataFrame:
    # Load the data
    data = pd.read_csv(data_path, header=0, sep=sep)
    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    return df


def parse_groundtruth(groundtruth: str) -> dict:
    # Split the string over colon
    split = groundtruth.split(":")
    keys = []
    values = []
    for words in split[:-1]:
        keys.append(words.split()[-1])

    # split groundtruth over the keys
    for i in range(len(keys)):
        if i == len(keys) - 1:
            values.append(split[i + 1].split())
        else:
            values.append(split[i + 1].split()[:-1])

    # Convert the values list to a string
    values = [" ".join(x) for x in values]

    # Assert that the length of the keys and values lists are equal
    assert len(keys) == len(values)

    # Create a dictionary using key-value pairs from the split list
    data_dict = {key: value for key, value in zip(keys, values)}
    return data_dict


def parse_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    for i, row in df.iterrows():
        groundtruth = row[col]
        data_dict = parse_groundtruth(groundtruth)

        # Add the dict to the dataframe
        for key, value in data_dict.items():
            df.loc[i, key] = value

    # Drop the original column
    df = df.drop(columns=[col])
    return df


def expand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Parse the groundtruth column
    df = parse_column(df, "groundtruth")
    df = parse_column(df, "prediction")

    return df


def process_gender_transcript(table: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate rows
    table = table.drop_duplicates()

    # Remove rows with missing values
    table = table.dropna()

    # Add empty column "sex'
    table["sex"] = None

    # Iterate over the rows
    for i, row in table.iterrows():
        # Get the number of male and female with the firstname
        n_male, n_female = row["male"], row["female"]

        # Add the maximum of to the column "sex"
        if n_male > n_female:
            table.iloc[i, -1] = "male"
        else:
            table.iloc[i, -1] = "female"

    # Drop the "male" and "female" columns
    table = table.drop(columns=["male", "female"])

    return table


def main() -> None:
    transcription_path = Path("data/transcriptions_with_sex.csv")
    firstname_sex_path = Path("data/firstname_with_sex.csv")

    # Check if the paths exist
    if not transcription_path.exists():
        raise FileNotFoundError(f"File not found: {transcription_path}")
    if not firstname_sex_path.exists():
        raise FileNotFoundError(f"File not found: {firstname_sex_path}")

    # Load the data
    df = load_data(data_path=transcription_path)
    # Preprocess the data
    processed_df = preprocess_data(df)

    # Expand the dataset
    expanded_df = expand_dataset(processed_df)

    # Save the expanded dataset
    expanded_df.to_csv("data/preprocessed_dataset.csv", index=False)

    # Process the transcript data
    gender_table = load_data(data_path=firstname_sex_path, sep=";")
    gender_table = process_gender_transcript(table=gender_table)

    # Save the processed gender table
    gender_table.to_csv("data/gender_table.csv", index=False)


if __name__ == "__main__":
    main()
