import argparse
from datetime import datetime
from show_figure import show_figure, show_figure_month
from form_and_filter import form_dataframe, filter_by_deviation, filter_by_date, group_by_month


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get statistics on a csv file')

    parser.add_argument('--start_date',
                            type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                            help='Start date')
    parser.add_argument('--end_date',
                            type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                            help='End date')

    deviation_group = parser.add_mutually_exclusive_group()
    deviation_group.add_argument('--deviation_value',
                                 type=float,
                                 help='Deviation value for filter')

    target_month_group = parser.add_mutually_exclusive_group()
    target_month_group.add_argument('--target_month',
                                    type=str,
                                    help='Target month for group')

    parser.add_argument('--path_file',
                        type=str, default="dataset/dataset.csv",
                        help='the path to the csv file with the data')
    parser.add_argument('--new_path',
                        type=str, default="dataset/stat.csv",
                        help='new path of modified csv')
    
    args = parser.parse_args()
    df = form_dataframe(args.path_file)

    if args.start_date and args.end_date:
        date_df = filter_by_date(df, args.start_date, args.end_date)
        show_figure(date_df)
    elif args.deviation_value:
        deviation_df = filter_by_deviation(df, args.deviation_value)
        show_figure(deviation_df)
    elif args.target_month:
        month_df, monthly_avg = group_by_month(df)
        show_figure_month(month_df, args.target_month)