from cv2 import dft
import pandas as pd

def open_new_csv(csv_path:str) -> pd.DataFrame:
    df=pd.read_csv(csv_path, delimiter=',', names=['Absolute path', 'Relative path', 'Tag'])
    df=df.drop("Relative path",axis=1)
    return df

def open_csv(csv_path:str) -> pd.DataFrame:
    df=pd.read_csv(csv_path)
    return df

def save_csv(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path,index=False)
