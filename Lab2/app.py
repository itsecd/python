import argparse
from write_reader import *       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('maindir', help='directory from which the search will be performed', type=str)
    parser.add_argument('-c', '--copy', help='''copies the contents of the directory in the selected format: 
                                                'num_class' - 'xxx//class_label_0xxx.jpg'
                                                'rand_num' - 'xxx//0xxx.jpg''',
                                                type=str, default=None)
    parser.add_argument('-cp','--csv_path', help='path where the csv file will be saved', type=str, default='data.csv')
    parser.add_argument('-dp','--data_path', help='path where the datafiles will be saved if a copy is made', type=str, default='dataset1')
    args = parser.parse_args() 

    dataset = DataWriteReader(args.maindir)
    if args.copy:
        if args.copy == 'num_class':
            print('num_class')
        elif args.copy == 'rand_num':    
            print('rand_num')
        else:
            logging.error(f"'{args.copy}' - unknown value of parameter 'COPY'")    
    print('write_to_csv')