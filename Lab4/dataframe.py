import pandas as pd
import logging
import json
import cv2
import os

#python dataframe.py
logging.basicConfig(filename="log.log", filemode="a", level=logging.INFO)

def create_dataframe(csv_path: str, csv_filename: str, tag: str) -> None:
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
    csv_file['Channels'] = depth_img
    csv_file['Label'] = labels
    csv_file.to_csv(csv_filename, index=False)
    print(csv_file)

if __name__=="__main__":
    with open('settings.json','r') as f:
        settings=json.load(f)
    create_dataframe(os.path.join(r'C:\Users\Yana\Documents\python-v6\Lab2', 'file.csv'), 'data.csv', settings['tag_tiger'])
   
