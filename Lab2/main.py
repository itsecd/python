import os
import X_Y_csv
import year_csv
import week_csv
from create_folder import create_folder
from get_data import input_csv
from get_data import get_data_for_date

def main_function(args: dict) -> None:
    if args['mode'] == "X_Y":
        create_folder(args['path_file'])
        output_csv_x = os.path.join(args['path_file'], args['output_x'])
        output_csv_y = os.path.join(args['path_file'], args['output_y'])
        X_Y_csv.split_csv(input_csv, output_csv_x, output_csv_y)
    elif args['mode'] == "years":
        create_folder(args['path_file'])
        year_csv.split_by_year(input_csv, args['path_file'])
    elif args['mode'] == "weeks":
        create_folder(args['path_file'])
        week_csv.split_by_week(input_csv, args['path_file'])
    elif args['mode'] == "find data":
        data_for_date = get_data_for_date(args['date'], input_csv, args['output_x'], args['output_y'])
        print(f"Value for {args['date']}: {data_for_date}")