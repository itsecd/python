import pandas as pd


def open_new_csv(csv_path: str) -> pd.DataFrame:
    """The function opens the csv as a dataframe 
    naming the columns"""
    df = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Tag"]
    )
    df = df.drop("Relative path", axis=1)
    return df


def open_csv(csv_path: str) -> pd.DataFrame:
    """The function opens csv as dataframe"""
    df = pd.read_csv(csv_path)
    return df


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """The function saves the dataframe to csv"""
    df.to_csv(file_path, index=False)
