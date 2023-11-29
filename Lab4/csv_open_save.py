import pandas as pd

def open_csv(csv_path : str) -> pd.DataFrame:
    dframe = pd.read_csv(csv_path,
                         delimiter=',',
                         names=["Absolute path", "Relative path", "Class"])
    dframe = dframe.drop("Relative path", axis=1, inplace=True)
    return dframe

def save_csv(dframe : pd.DataFrame, path : str) -> None:
    dframe.to_csv(path)