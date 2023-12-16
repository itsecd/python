import pandas as pd

def open_csv(csv_path: str, original=True) -> pd.DataFrame:
    '''
    Opens a CSV file as a dataframe.
    If original is True, it assigns column names and removes unnecessary columns.
    '''
    if original:
        dataframe = pd.read_csv(
            csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
        )
        dataframe = dataframe.drop("Relative path", axis=1)
    else:
        dataframe = pd.read_csv(csv_path)

    return dataframe

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    '''
    Saves a dataframe to a CSV file.
    '''
    dataframe.to_csv(file_path, index=False)
