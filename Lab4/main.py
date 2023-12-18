import argparse
from read_and_write import open_csv, save_csv
import functions
import the_graphic_part
import logging


logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function to handle command-line arguments and execute corresponding functions.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--create_dataframe", action="store_true",
                       help="Create DataFrame")
    group.add_argument("--compute_statistics", action="store_true",
                       help="Compute statistics")
    group.add_argument("--filter_by_label", action="store_true",
                       help="Filter DataFrame by label")
    group.add_argument("--filter_by_params", action="store_true",
                       help="Filter DataFrame by parameters")
    group.add_argument("--group_by_label_and_pixel_count", action="store_true",
                       help="Group DataFrame by label and pixel count")
    group.add_argument("--create_histogram", action="store_true",
                       help="Create histogram")

    parser.add_argument("--csv_path", type=str, help="Path to input CSV file")
    parser.add_argument("--new_file_path", type=str,
                        help="Path to save the result CSV file")

    args = parser.parse_args()

    try:
        if args.create_dataframe:
            df = functions.create_dataframe(args.csv_path)
            save_csv(df, args.new_file_path)
            logging.info("DataFrame created and saved.")

        elif args.compute_statistics:
            df = functions.create_dataframe(args.csv_path)
            statistics_df = functions.compute_statistics(df)
            save_csv(statistics_df, args.new_file_path)
            logging.info("Statistics computed and saved.")

        elif args.filter_by_label:
            df = open_csv(args.csv_path)
            filter_df = functions.filter_dataframe(
                df, target_label=1, max_height=None, max_width=None)
            save_csv(filter_df, args.new_file_path)
            logging.info("Filtered DataFrame by label and saved.")

        elif args.filter_by_params:
            df = open_csv(args.csv_path)
            filter_df = functions.filter_dataframe(
                df, target_label=1, max_height=100, max_width=100)
            save_csv(filter_df, args.new_file_path)
            logging.info("Filtered DataFrame by parameters and saved.")

        elif args.group_by_label_and_pixel_count:
            df = open_csv(args.csv_path)
            grouped_df = functions.group_by_label_and_pixel_count(df)
            save_csv(grouped_df, args.new_file_path)
            logging.info(
                "Grouped DataFrame by label and pixel count, and saved.")

        elif args.create_histogram:
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
