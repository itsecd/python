from turtle import width
from csv_open_save import open_csv,  save_csv
import dataframe
import argparse
import logging
import sys

logging.basicConfig(filename="log.log", filemode="a", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument(
        "-o",
        "--option",
        type=int,
        default=5,
        choices=range(5),
        help="0 - Test for balance"
        "1 - Filter by label"
        "2 - Filter by label with parameters"
        "3 - Groupping"
        "4 - Make histogram by random image",
    )

    parser.add_argument("--csv_path", help="Path to csv_file")
    parser.add_argument("--label", type=int, help="Label of image")
    parser.add_argument("--width", type=int, help="Width of image")
    parser.add_argument("--height", type=int, help="Height of image")
    parser.add_argument("--csv_save", help="Path to save file")
    args = parser.parse_args()
    if args.option == 0:
        df = dataframe.make_dataframe(
            open_csv(
                args.csv_path,
                column_names=["Absolute path", "Relative path", "Tag"],
                remove_column="Relative path",
            ),
            "tiger",
        )
        save_csv(dataframe.make_stats(df), args.csv_save)
        logging.info("Test for balance work")
    elif args.option == 1:
        df = open_csv(args.csv_path)
        logging.info(dataframe.filter_by_label(df, args.label))
        logging.info("Filter by label work")
    elif args.option == 2:
        df = open_csv(args.csv_path)
        logging.info(
            dataframe.filter_with_param(df, args.width, args.height, args.label)
        )
        logging.info("Filter by label with param work")
    elif args.option == 3:
        df = open_csv(args.csv_path)
        logging.info(dataframe.groupping(df))
        logging.info("Groupping work")
    elif args.option == 4:
        df = open_csv(args.csv_path)
        hists = dataframe.make_histogram(df, args.label)
        dataframe.draw_histogram(hists)
    else:
        logging.warning("Option was not selected")


if __name__ == "__main__":
    sys.exit(main())
