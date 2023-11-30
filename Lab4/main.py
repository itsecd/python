from functions import (make_dframe,
                       make_stats,
                       filter_by_type,
                       filter_by_size,
                       grouping,
                       make_hists,
                       draw_hists)
from csv_open_save import open_csv, save_csv
import os
import argparse

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
    parser.add_argument("-h", "--height", type=int, help="Height of image")
    parser.add_argument("-c", "--class", help="Class of image (rose or tulip)")
    parser.add_argument("-n", "--new_file_path", help="Path to save")
    
    args = parser.parse_args()

    match args.option:
        case 0:
            dfame = make_dframe(open_csv(args.csv_path), "rose")
            save_csv(make_stats(dfame), args.new_file_path)
        case 1:
            dframe = open_csv(args.csv_path)
            print(filter_by_type(dframe, args.type))
        case 2:
            dframe = open_csv(args.csv_path)
            print(filter_by_size(dframe, args.width, args.height, args.type))
        case 3:
            dframe = open_csv(args.csv_path)
            print(grouping(dframe))
        case 4:
            dframe = open_csv(args.csv_path)
            draw_hists(make_hists(dframe, args.type))
        case _:
            print("You dont choose any option!\nBye!")

            
