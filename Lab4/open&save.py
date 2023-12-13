import pandas as pd

def read_csv_to_dataframe(file_path: str):
    """
    Function for opening a CSV file and creating a Data Frame
    """
    dataframe = pd.read_csv(file_path)
    return dataframe
    

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Function for saving the Data Frame to a CSV file.
    """
    dataframe.to_csv(file_path, index=False)
  

