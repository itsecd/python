import pandas as pd


def create_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Function for creating a DataFrame from a CSV file.

    Parameters:
    - csv_path: The path to the CSV file.
    Returns:
    - pd.DataFrame: The created DataFrame with columns 'class' and 'absolute_path'.
    """
    df = pd.read_csv(csv_path, header=None, names=[
                     'absolute_path', 'relative_path', 'class'], sep=',')
    result_df = df[['class', 'absolute_path']].copy()
    result_df.columns = ['class', 'absolute_path']
    return result_df


def open_csv(csv_path: str) -> pd.DataFrame:
    """
    Function for opening the CSV annotation as a DataFrame.

    Parameters:
    - csv_path: The path to the CSV file.
    Returns:
    - pd.DataFrame: The loaded DataFrame from the CSV file.
    """
    return pd.read_csv(csv_path)


def save_csv(dframe: pd.DataFrame, file_path: str) -> None:
    """
    Function for saving a DataFrame to a CSV file .

    Parameters:
    - dframe: The DataFrame to be saved.
    - file_path: The name of the CSV file where the DataFrame will be saved.
    """
    dframe.to_csv(file_path, index=False)
