from functions import (make_dframe,
                       make_stats,
                       filter_by_type,
                       filter_by_size,
                       grouping,
                       make_hists,
                       draw_hists)
from csv_open_save import open_csv, save_csv, open_new_csv
import argparse
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Choosing mode")
    parser.add_argument('-o', '--option', type=int, default=5, help='0 - Test for balance'
                                                                    '1 - Filter by type'
                                                                    '2 - Filter by max height, width and type,'
                                                                    '3 - Grouping'
                                                                    '4 - Make histogram by random image')
    
    parser.add_argument("-p", "--csv_path", help="Path to csv file")
    parser.add_argument("-t", "--type", type=int, help="Type of image (0 or 1)")
    parser.add_argument("-w", "--width", type=int, help="Width of image")
    parser.add_argument("--height", type=int, help="Height of image")
    parser.add_argument("-c", "--class", help="Class of image (rose or tulip)")
    parser.add_argument("-n", "--new_file_path", help="Path to save")
    
    args = parser.parse_args()

    match args.option:
        case 0:
            dfame = make_dframe(open_new_csv(args.csv_path))
            save_csv(make_stats(dfame), args.new_file_path)
            logging.info("Balance check is successfull")
        case 1:
            dframe = make_dframe(args.csv_path)
            print(filter_by_type(dframe, args.type))
            logging.info("Filter by type is successfull")
        case 2:
            dframe = make_dframe(args.csv_path)
            print(filter_by_size(dframe, args.width, args.height, args.type))
            logging.info("Filter by size and type is successfull")
        case 3:
            dframe = open_csv(args.csv_path)
            print(grouping(dframe))
            logging.info("Grouping is successfull")
        case 4:
            dframe = open_csv(args.csv_path)
            draw_hists(make_hists(dframe, args.type))
            logging.info("Draw histogram is successfull")
        case _:
            logging.warning("You dont choose any option!")

            
