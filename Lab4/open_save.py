import pandas as pd


def read_csv_to_dataframe(csv_file: str) -> pd.DataFrame:
    '''function for reading a csv file in a dataframe'''
    return pd.read_csv(csv_file, delimiter=";", names=["Absolute path", "Relative path", "Class"])


def save_to_csv(df: pd.DataFrame, output_csv: str) -> None:
    '''function for saving the dataframe to a csv file'''
    df.to_csv(output_csv, index=False, sep=';')