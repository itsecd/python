import argparse
from read_and_write import open_csv, save_csv
import functions

def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--function1", action="store_true", help="create Dataframe")
    group.add_argument("--function2", action="store_true", help="compute statistics")
    group.add_argument("--function3", action="store_true", help="other function")

    parser.add_argument("--csv_path", type=str, help="path to input CSV file")
    parser.add_argument("--new_file_path", type=str, help="path to save the result CSV file")

    args = parser.parse_args()

    if args.function1:
        df = functions.create_dataframe(args.csv_path)
        save_csv(df, args.new_file_path)

    elif args.function2:
        df = functions.create_dataframe(args.csv_path)
        statistics_df = functions.compute_statistics(df)
        save_csv(statistics_df, args.new_file_path)
        
    elif args.function3:
        # for the next function
        return

if __name__ == "__main__":
    main()