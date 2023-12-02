import pandas as pd



def open_new_csv(csv_path : str) -> pd.DataFrame:
    """Function for open new file as a dataframe"""
    dframe = pd.read_csv(csv_path,
                         delimiter=',',
                         names=["Absolute path", "Relative path", "Class"])
    dframe = dframe.drop(["Relative path"], axis=1, inplace=True)
    return dframe

def open_csv(csv_path: str) -> pd.DataFrame:
    """Function for opening the csv-annotation as a dataframe"""
    dframe = pd.read_csv(csv_path)
    return dframe

def save_csv(dframe : pd.DataFrame, path : str) -> None:
    """
    Function for save dframe as a csv file
    """
    dframe.to_csv(path)