import pandas as pd


def open_csv(
    csv_path: str,
    delimiter: str = ",",
    column_names: list = None,
    remove_column: str = None,
) -> pd.DataFrame:
    """The function opens the csv as a dataframe with optional column names and delimiter"""
    if column_names is not None:
        df = pd.read_csv(csv_path, delimiter=delimiter, names=column_names)
    else:
        df = pd.read_csv(csv_path, delimiter=delimiter)

    if remove_column is not None:
        df = df.drop(remove_column, axis=1)

    return df


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """The function saves the dataframe to csv"""
    df.to_csv(file_path, index=False)
