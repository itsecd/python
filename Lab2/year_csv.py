import argparse
import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def create_folder(base_folder: str) -> None:
    """The function form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
    except Exception as ex:
        logging.exception(f"Can't create folder: {ex}\n{ex.args}\n")


def split_by_year(input_file: str,
                  output_path: str
                  ) -> None:
    """The function takes path to the input file and split file to years"""
    try:
        create_folder(output_path)
        df = pd.read_csv(input_file, names=['Date','Value'])

        df['Date'] = pd.to_datetime(df['Date'])

        grouped = df.groupby(df['Date'].dt.year)

        for year, group in grouped:
            output_file = f"{group['Date'].dt.strftime('%Y%m%d').min()}_{group['Date'].dt.strftime('%Y%m%d').max()}.csv"

            group.to_csv(os.path.join(output_path,output_file), index=False)
    except Exception as ex:
        logging.exception(f"Can't split data to years: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for years.')
    parser.add_argument('--path_file',
                        type=str, default='years',
                        help='The path to the data file'
                        )
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    split_by_year(input_csv,args.path_file)