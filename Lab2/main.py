import argparse
import os
import X_Y_csv
import year_csv
import week_csv
from datetime import datetime
from create_folder import create_folder
from get_data import input_csv
from get_data import get_data_for_date

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split csv file for weeks.')
    parser.add_argument('--mode',
                        type=str, default='find data',
                        help='mode selection'
                        )
    parser.add_argument('--path_file',
                        type=str, default='weeks',
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
    parser.add_argument('--date',
                        type=datetime, default=datetime(2023,10,5),
                        help='Output file name Y'
                        )

    args = parser.parse_args()
    if (args.mode == "X_Y"):
        create_folder(args.path_file)
        output_csv_x = os.path.join(args.path_file,args.output_x)
        output_csv_y = os.path.join(args.path_file,args.output_y)
        X_Y_csv.split_csv(input_csv,output_csv_x,output_csv_y)
    if (args.mode == "years"):
        create_folder(args.path_file)
        year_csv.split_by_year(input_csv,args.path_file)
    if (args.mode == "weeks"):
        create_folder(args.path_file)
        week_csv.split_by_week(input_csv,args.path_file)
    if (args.mode == "find data"):
        data_for_date = get_data_for_date(args.date,input_csv,args.output_x, args.output_y)
        print(f"Value for: {args.date}: {data_for_date}")