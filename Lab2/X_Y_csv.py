import argparse
import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)


def create_folder(base_folder: str = "X_and_Y") -> None:
    """The function form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
    except Exception as ex:
        logging.exception(f"Can't create folder: {ex.message}\n{ex.args}\n")


def split_csv(input_file: str,
              output_file_x: str,
              output_file_y: str
              ) -> None:
    """The function takes path to the input file, name of output file x
    and name of output file y and split file to date and data"""
    try:
        df = pd.read_csv(input_file)

        df_x = df.iloc[:, 0]
        df_y = df.iloc[:, 1]

        df_x.to_csv(output_file_x, index=False, header=False)
        df_y.to_csv(output_file_y, index=False, header=False)
    except Exception as ex:
        logging.exception(f"Can't split csv to X and Y: {ex}\n{ex.args}\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file to dates and data.')
    parser.add_argument('--path_file',
                        type=str, default='X_and_Y',
                        help='The path to the data file'
                        )
    parser.add_argument('--outputX',
                        type=str, default='X.csv',
                        help='Output file name X'
                        )
    parser.add_argument('--outputY',
                        type=str, default='Y.csv',
                        help='Output file name Y'
                        )
    
    args = parser.parse_args()

    create_folder(args.path_file)
    output_csv_x = os.path.join(args.path_file,args.outputX)
    output_csv_y = os.path.join(args.path_file,args.outputY)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    try:
        if os.path.exists(input_csv):
            split_csv(input_csv, output_csv_x, output_csv_y)
            logging.info("files successfully created.")
    except Exception as ex:
        logging.exception(f"Can't create files: {ex}\n{ex.args}\n") 