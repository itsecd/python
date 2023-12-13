import argparse
from read_and_write import open_csv, save_csv
import functions
import the_graphic_part
import logging


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--function1", action="store_true",
                       help="create Dataframe")
    group.add_argument("--function2", action="store_true",
                       help="compute statistics")
    group.add_argument("--function3", action="store_true",
                       help="filter_dataframe_by_label")
    group.add_argument("--function4", action="store_true",
                       help="filter_dataframe_by_params")
    group.add_argument("--function5", action="store_true",
                       help="group_by_label_and_pixel_count")
    group.add_argument("--function6", action="store_true",
                       help="create histogram")

    parser.add_argument("--csv_path", type=str, help="path to input CSV file")
    parser.add_argument("--new_file_path", type=str,
                        help="path to save the result CSV file")

    args = parser.parse_args()

    try:
        if args.function1:
            df = functions.create_dataframe(args.csv_path)
            save_csv(df, args.new_file_path)
            logging.info("Dataframe created and saved.")

        elif args.function2:
            df = functions.create_dataframe(args.csv_path)
            statistics_df = functions.compute_statistics(df)
            save_csv(statistics_df, args.new_file_path)
            logging.info("Statistics computed and saved.")

        elif args.function3:
            df = open_csv(args.csv_path)
            filter_df = functions.filter_dataframe(
                df, target_label=1, max_height=None, max_width=None)
            save_csv(filter_df, args.new_file_path)
            logging.info("Filtered dataframe by label and saved.")

        elif args.function4:
            df = open_csv(args.csv_path)
            filter_df = functions.filter_dataframe(
                df, target_label=1, max_height=100, max_width=100)
            save_csv(filter_df, args.new_file_path)
            logging.info("Filtered dataframe by params and saved.")

        elif args.function5:
            df = open_csv(args.csv_path)
            grouped_df = functions.group_by_label_and_pixel_count(df)
            save_csv(grouped_df, args.new_file_path)
            logging.info(
                "Grouped dataframe by label and pixel count, and saved.")

        elif args.function6:
            df = open_csv(args.csv_path)
            target_label = 1
            hist_b, hist_g, hist_r = the_graphic_part.calculate_opencv_histogram(
                df, target_label)
            the_graphic_part.plot_opencv_histogram(
                hist_b, hist_g, hist_r, target_label)
            the_graphic_part.plot_histogram_matplotlib(
                hist_b, hist_g, hist_r, target_label)
            logging.info("Histograms created and plotted.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
