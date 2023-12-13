import img_df
import img_hist
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stats", action="store_true", help="calc statistic-data for attrs as like: width, height, channels, class")
    parser.add_argument("--filter", action="store_true", help="filtered selection by class label")
    parser.add_argument("--group", action="store_true", help="sample based on the specified characteristics")
    parser.add_argument("--hist", action="store_true", help="builds a histogram based on the specified values")
    parser.add_argument("--save", action="store_true", help="save df to csv")
    parser.add_argument('csv_path', help='base csv path', type=str)
    parser.add_argument('-df','--df_path', help='path where the csv file with df will be saved', type=str, default='data.csv')
    parser.add_argument('-st','--stats_path', help='path where the csv file with stats will be saved', type=str, default='stats.csv')
    parser.add_argument('-mw','--max_width', help='width param for filter', type=int, default=None)
    parser.add_argument('-mh','--max_height', help='height param for filter', type=int, default=None)
    parser.add_argument('-c','--class_label', help='class label param for filter', type=int, default=None)    

    args = parser.parse_args()

    df = img_df.df_from_csv(args.csv_path)
    if args.stats:
        stats = img_df.df_imgs_stats(df)
        print(stats)
        if args.save:
            stats.to_csv(args.stats_path, index=False)
    if args.filter:
        df = img_df.filter_by(df, class_lbl=args.class_label, wmax=args.max_width, hmax=args.max_height)
        print(df)    
    if args.group:
        df = img_df.group_by_resolution(df)
        print(df)
    if args.hist:
        img_hist.draw_hist(img_df.build_hist(df,args.class_label))
    if args.save:
        df.to_csv(args.df_path, index=False)  

if __name__ == "__main__":
    main()