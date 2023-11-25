import argparse
import pandas as pd


def form_dataframe(file_path: str,
                   new_name: str
                   ) -> None:
    df = pd.read_csv(file_path,delimiter=",")
    df.columns = ['Дата', 'Курс']
    df['Курс'] = pd.to_numeric(df['Курс'], errors="coerce")
    
    invalid_values = df[df['Курс'].isnull()]
    if not invalid_values.empty:
        df = df.dropna(subset=['Курс'])
        
    median_value = df['Курс'].median()
    mean_value = df['Курс'].mean()
    df['Отклонение от медианы'] = df['Курс'] - median_value
    df['Отклонение от среднего'] = df['Курс'] - mean_value
    
    statistics = df[['Курс', 'Отклонение от медианы', 'Отклонение от среднего']].describe()
    print(statistics)
    
    df.to_csv(new_name, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='get statistics on a csv file')
    parser.add_argument('--path_file',
                        type=str, default="dataset/dataset.csv",
                        help='the path to the csv file with the data'
                        )
    parser.add_argument('--new_path',
                        type=str, default="dataset/stat.csv",
                        help='new path of modified csv'
                        )
    args = parser.parse_args()
    form_dataframe(args.path_file,args.new_path)
    