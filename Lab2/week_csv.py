import argparse
import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def create_folder(base_folder: str = "weeks") -> None:
    """The function form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
    except Exception as ex:
        logging.exception(f"Can't create folder: {ex}\n{ex.args}\n")


def split_by_week(input_file: str,
                  output_path
                  ) -> None:
    """The function takes path to the input file and split file to weeks"""
    try:
        create_folder(output_path)
        df = pd.read_csv(input_file, names=['Date','Value'])

        df['Date'] = pd.to_datetime(df['Date'])

        grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.isocalendar().week])

        for (year, week), group in grouped:
            start_date = group['Date'].min().strftime('%Y%m%d')
            end_date = group['Date'].max().strftime('%Y%m%d')

            output_file = f"{start_date}_{end_date}.csv"

            group.to_csv(os.path.join(output_path,output_file), index=False)
    except Exception as ex:
        logging.exception(f"Can't split data to weeks: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for weeks.')
    parser.add_argument('--path_file',
                        type=str, default='weeks',
                        help='The path to the data file'
                        )
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    split_by_week(input_csv,args.path_file)