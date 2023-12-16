import pandas as pd

def open_original_csv(csv_path: str) -> pd.DataFrame:
    '''
    Opens the original annotation CSV as a dataframe, assigns column names, and removes unnecessary columns.
    '''
    dataframe = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
    )
    dataframe = dataframe.drop("Relative path", axis=1)
    return dataframe

def open_csv_annotation(csv_path: str) -> pd.DataFrame:
    '''
    Opens the CSV annotation file as a dataframe.
    '''
    dataframe = pd.read_csv(csv_path)
    return dataframe

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    '''
    Saves a dataframe to a CSV file.
    '''
    dataframe.to_csv(file_path, index=False)
