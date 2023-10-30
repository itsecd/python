import argparse
from write_reader import *       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('maindir', help='directory from which the search will be performed', type=str)
    parser.add_argument('-cp','--csv_path', help='path where the csv file will be saved', type=str, default='data.csv')
    parser.add_argument('-dp','--data_path', help='path where the csv file will be saved', type=str, default='dataset1')
    args = parser.parse_args() 

    dataset = DataWriteReader(args.maindir)  
    dataset2 = dataset.copy_dataset_randnum(args.data_path)
    dataset2.write_to_csv(args.csv_path)