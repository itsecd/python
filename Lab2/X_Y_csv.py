import argparse
import os
import logging
import csv
from datetime import datetime
from create_folder import create_folder


logging.basicConfig(level=logging.INFO)


def split_csv(input_file: str,
              output_file_x: str,
              output_file_y: str
              ) -> None:
    """The function takes path to the input file, name of output file x
    and name of output file y and split file to date and data"""
    try:
        with open(input_file, 'r') as infile, \
             open(output_file_x, 'w', newline='') as outfile_x, \
             open(output_file_y, 'w', newline='') as outfile_y:
            
            reader = csv.reader(infile)
            writer_x = csv.writer(outfile_x)
            writer_y = csv.writer(outfile_y)

            header = next(reader)
            x_index = 0
            y_index = 1

            writer_x.writerow([header[x_index]])
            writer_y.writerow([header[y_index]])

            for row in reader:
                writer_x.writerow([row[x_index]])
                writer_y.writerow([row[y_index]])
    except Exception as ex:
        logging.exception(f"Can't split csv to X and Y: {ex}\n{ex.args}\n")


def read_data_from_x_y(date: datetime,
                       dates_file_path: str,
                       values_file_path: str
                       ) -> str:
    """The function takes the date for which the data needs to be found,
    the paths to the files x and y, and returns the data"""
    data = None
    try:
        if os.path.exists(dates_file_path) and os.path.exists(values_file_path):
            with open(dates_file_path, 'r') as dates_file, \
                 open(values_file_path, 'r') as values_file:
                
                dates_reader = csv.reader(dates_file)
                values_reader = csv.reader(values_file)

                header_dates = next(dates_reader)
                header_values = next(values_reader)

                date_index = header_dates.index('Date')
                value_index = header_values.index('Value')

                date_to_compare = date.date()
                
                for row in dates_reader:
                    row_date = datetime.strptime(row[date_index], '%Y-%m-%d').date()
                    value_row = next(values_reader)
                    if row_date == date_to_compare:
                        data = value_row[value_index]
                        if data == "data not found":
                            data = None
                        break
        return data
    except Exception as ex:
        logging.exception(f"Can't read data from x and y files: {ex}\n{ex.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file to dates and data.')
    parser.add_argument('--path_file',
                        type=str, default='X_and_Y',
                        help='The path to the data file'
                        )
    parser.add_argument('--output_x',
                        type=str, default='X.csv',
                        help='Output file name X'
                        )
    parser.add_argument('--output_y',
                        type=str, default='Y.csv',
                        help='Output file name Y'
                        )
    
    args = parser.parse_args()

    create_folder(args.path_file)
    output_csv_x = os.path.join(args.path_file,args.output_x)
    output_csv_y = os.path.join(args.path_file,args.output_y)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "../Lab1/dataset/dataset.csv")

    split_csv(input_csv, output_csv_x, output_csv_y)