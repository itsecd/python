import pandas as pd


def open_new_csv(csv_path: str) -> pd.DataFrame:
    '''Function for opening the original annotation as a dataframe,
    naming columns of the dataframe and removing unnecessary'''
    df = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
    )
    df = df.drop("Relative path", axis=1)
    return df


def open_csv(csv_path: str) -> pd.DataFrame:
    '''Function for opening the csv-annotation as a dataframe'''
    df = pd.read_csv(csv_path)
    return df


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    '''Function of saving a dataframe to a csv file'''
    df.to_csv(file_path, index=False, encoding="utf-8")