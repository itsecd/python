import argparse
import functions
from histogram import plot_histograms
from open_save_script import open_original_csv, open_csv_annotation, save_dataframe_to_csv

def main():
    """
    Main function that parses command line arguments and executes the selected option.
    """
    parser = argparse.ArgumentParser(description="Image Dataset Analysis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--balance-check", action="store_true", help="Check dataset balance")
    group.add_argument("--filter-by-label", action="store_true", help="Filter DataFrame by label")
    group.add_argument("--filter-by-min-max", action="store_true", help="Filter DataFrame by min-max values")
    group.add_argument("--group-by-label", action="store_true", help="Group DataFrame by label")
    group.add_argument("--create-histogram", action="store_true", help="Create histogram for a random image")
    parser.add_argument("--csv-path", help="Path to the base CSV file")
    parser.add_argument("--label", type=int, help="Label of the image (0 or 1)")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    parser.add_argument("--class", help="Class of the image (rose or tulip)")
    parser.add_argument("--new-file-path", help="Path for saving the new CSV file")
    args = parser.parse_args()
    if args.balance_check:
        dataframe = functions.add_image_parameters(open_original_csv(args.csv_path), "rose")
        open_csv_annotation(functions.check_balance(dataframe), args.new_file_path)
    elif args.filter_by_label:
        dataframe = save_dataframe_to_csv(args.csv_path)
        print(functions.filter_by_label(dataframe, args.label))
    elif args.filter_by_min_max:
        dataframe = save_dataframe_to_csv(args.csv_path)
        print(functions.min_max_filter(dataframe, args.width, args.height, args.label))
    elif args.group_by_label:
        dataframe = save_dataframe_to_csv(args.csv_path)
        print(functions.group_by_label(dataframe))
    elif args.create_histogram:
        dataframe = save_dataframe_to_csv(args.csv_path)
        plot_histograms(functions.build_histogram(dataframe, args.label))
    else:
        print("No option selected")

if __name__ == "__main__":
    main()
