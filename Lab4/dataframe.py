import pandas as pd
import logging
import json
import cv2
import os

#python dataframe.py
logging.basicConfig(filename="log.log", filemode="a", level=logging.INFO)

def create_dataframe(csv_path: str, csv_filename: str, tag: str) -> pd.DataFrame:
    print('in function')
    csv_file = pd.read_csv(csv_path, delimiter=',', names=['Absolute path', 'Relative path', 'Tag'])
    abs_path = csv_file['Absolute path']
    tags = csv_file['Tag']
    height_img = []
    width_img = []
    depth_img = []
    labels = []
    count = 0
    try:
        for img in abs_path:
            path = cv2.imread(img)
            tuple = path.shape
            height_img.append(tuple[0])
            width_img.append(tuple[1])
            depth_img.append(tuple[2])
            if tags[count] == tag:
                label = 0
            else:
                label = 1
            labels.append(label)
            count += 1
    except Exception as e:
        logging.error(f'Error get img by idx{e}')
    csv_file = csv_file.drop('Relative path', axis=1)
    csv_file['Height'] = height_img
    csv_file['Width'] = width_img
    csv_file['Depth'] = depth_img
    csv_file['Label'] = labels
    csv_file.to_csv(csv_filename, index=False)
    return csv_file

def balance(df_file: str, st_file: str) -> pd.DataFrame:
    csv_file=pd.read_csv(df_file)
    img=csv_file[["Height","Width","Depth"]].describe()
    label_st=csv_file['Label']
    label_info=label_st.value_counts()
    df=pd.DataFrame()
    tag_mentions=label_info.values
    balance=tag_mentions[0]/tag_mentions[1]
    df["Labels"]=label_info
    df["Tag mentions"]=tag_mentions
    df['Balance']=f'{balance:.1f}'
    pd.concat([img,df],axis=1).to_csv(st_file)
    if (balance>=0.95 and balance<=1.05):
        logging.info("Balanced")
    else:
        logging.info(f"Not balanced,{abs(balance*100-100):.1f}%")
    return df

def filter_by_label(df:pd.DataFrame,label:int) -> pd.DataFrame:
    filtered_df=df[df["Label"]==label]
    return filtered_df

def filter_with_param(df:pd.DataFrame,width_max:int,height_max:int,label:str) -> pd.DataFrame:
    filtered_df=df[(df["Label"]==label)& (df["Width"]<=width_max)&(df["Height"]<=height_max)]
    return filtered_df

if __name__=="__main__":
    with open('settings.json','r') as f:
        settings=json.load(f)
    #create_dataframe(os.path.join(r'C:\Users\Yana\Documents\python-v6\Lab2', 'file.csv'), 'data.csv', settings['tag_tiger'])
    csv_file=pd.read_csv("data.csv")
    #balance('data.csv','statistic.csv')
    filter=filter_with_param(csv_file,180,200,1)
    logging.info(filter)
    by_label=filter_by_label(csv_file,0)
    logging.info(by_label)


   
