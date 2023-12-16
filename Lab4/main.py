import argparse
import logging
from functions import add_image_parameters, check_balance, filter_by_label, filter_by_min_max, group_by_label, build_histogram
from histogram import plot_histograms
from open_save_script import open_csv, save_dataframe_to_csv

logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function that parses command line arguments and executes the selected option.
    """
    parser = argparse.ArgumentParser(description="Image Dataset Analysis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-balance", action="store_true", help="Check dataset balance")
    group.add_argument("--filter-by-label", action="store_true", help="Filter DataFrame by label")
    group.add_argument("--filter-by-min-max", action="store_true", help="Filter DataFrame by min-max values")
    group.add_argument("--group-by-label", action="store_true", help="Group DataFrame by label")
    group.add_argument("--build-histogram", action="store_true", help="Build histogram for a random image")
    parser.add_argument("--csv-path", help="Path to the base CSV file")
    parser.add_argument("--label", type=int, help="Label of the image (0 or 1)")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    parser.add_argument("--class", help="Class of the image (rose or tulip)")
    parser.add_argument("--new-file-path", help="Path for saving the new CSV file")
    args = parser.parse_args()

    if args.check_balance:
        dataframe = add_image_parameters(open_csv(args.csv_path), "rose")
        save_dataframe_to_csv(check_balance(dataframe), args.new_file_path)
    elif args.filter_by_label:
        dataframe = open_csv(args.csv_path, original=False)
        logging.info(filter_by_label(dataframe, args.label))
    elif args.filter_by_min_max:
        dataframe = open_csv(args.csv_path, original=False)
        logging.info(filter_by_min_max(dataframe, args.width, args.height, args.label))
    elif args.group_by_label:
        dataframe = open_csv(args.csv_path, original=False)
        logging.info(group_by_label(dataframe))
    elif args.build_histogram:
        dataframe = open_csv(args.csv_path, original=False)
        plot_histograms(build_histogram(dataframe, args.label))
    else:
        logging.info("No option selected")

if __name__ == "__main__":
    main()
