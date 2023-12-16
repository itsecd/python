import pandas as pd

def open_csv(csv_path: str, new_format: bool = False) -> pd.DataFrame:
    '''
    Function for opening the annotation CSV as a dataframe.
    If new_format is True, it renames columns and removes unnecessary ones.
    '''
    if new_format:
        dframe = pd.read_csv(
            csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
        )
        dframe.drop("Relative path", axis=1, inplace=True)
    else:
        dframe = pd.read_csv(csv_path)

    return dframe

def save_csv(dframe: pd.DataFrame, file_path: str) -> None:
    '''Function for saving a dataframe to a CSV file'''
    dframe.to_csv(file_path, index=False)