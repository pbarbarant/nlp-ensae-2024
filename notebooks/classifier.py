import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def loadData(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def matchFirstname(firstname: str, table: pd.DataFrame) -> str:
    # Find the corresponding row in the table with the firstname via fuzzy matching
    choices = table["firstname"].tolist()
    match = process.extractOne(firstname, choices, scorer=fuzz.WRatio)
    if match is None:
        return None
    else:
        selected_row = table[table["firstname"].str.contains(match[0])]
        return selected_row["sex"].values[0]


def predictSex(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    # Init prediction column
    df["prediction"] = None
    for i, row in df.iterrows():
        # Get the firstname
        firstname = (
            str(row["firstname"]).lower().strip()
            if not row["firstname"] == "nan"
            else str(row["prénom"]).lower().strip()
        )

        # Find the corresponding row in the table with the firstname via fuzzy matching
        sex = matchFirstname(firstname, table)
        # Add the prediction to the last column
        df.iloc[i, -1] = sex

    return df


def evalAccuracy(df: pd.DataFrame) -> float:
    counter = 0
    error_dict = {}
    for i, _ in df.iterrows():
        ground_truth = df["sex"].iloc[i]
        pred = df["prediction"].iloc[i]

        if ground_truth == pred:
            counter += 1
        else:
            firstname = str(df["firstname"].iloc[i]).lower().strip()
            if firstname in error_dict.keys():
                error_dict[firstname] += 1
            else:
                error_dict[firstname] = 1

    return counter / len(df), error_dict


def ROCCUrve(df: pd.DataFrame) -> tuple:
    le = LabelEncoder()
    # Remove rows with sex="ambigu"
    df = df[df["sex"] != "ambigu"]
    y_true = le.fit_transform(df["sex"])
    y_pred = le.transform(df["prediction"])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, metrics.RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name="Fuzzy Matching Classifier",
    )
