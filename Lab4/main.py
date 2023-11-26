import argparse
import pandas as pd
from datetime import datetime, date
from show_figure import show_figure, show_figure_month
from form_and_filter import form_dataframe, filter_by_deviation, filter_by_date, group_by_month


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get statistics on a csv file')
    parser.add_argument('--path_file',
                        type=str, default="dataset/dataset.csv",
                        help='the path to the csv file with the data'
                        )
    parser.add_argument('--new_path',
                        type=str, default="dataset/stat.csv",
                        help='new path of modified csv'
                        )
    parser.add_argument('--deviation_value',
                        type=float, default=20.0,
                        help='deviation value for filter'
                        )
    parser.add_argument('--start_date',
                        type=datetime, default=datetime(2023,9,1),
                        help='start date for filter'
                        )
    parser.add_argument('--end_date',
                        type=datetime, default=datetime(2023,10,1),
                        help='end date for filter'
                        )
    parser.add_argument('--target_month',
                        type=str, default="2023-09",
                        help='target month for group'
                        )
    args = parser.parse_args()
    df = form_dataframe(args.path_file)
    deviation_df = filter_by_deviation(df,args.deviation_value)
    date_df = filter_by_date(df,args.start_date,args.end_date)
    monthly_df, monthly_avg = group_by_month(df)
    show_figure(df)
    show_figure(deviation_df)
    show_figure(date_df)
    show_figure(monthly_df)
    show_figure_month(df,args.target_month)