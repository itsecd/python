import pandas as pd


def read_csv_to_dataframe(csv_file):
    return pd.read_csv(csv_file, delimiter=";", names=["Absolute path", "Relative path", "Class"])
def save_to_csv(df, output_csv):
    df.to_csv(output_csv, index=False, sep=';')