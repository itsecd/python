import pandas as pd
import logging
import cv2
import json

logging.basicConfig(level=logging.INFO)

def image_forms(csv_path:str, new_csv_filename:str, class_one) -> None:
        height_image = []
        width_image = []
        channels_image = []
        numerical = []
        counter = 0  # счетчик
        csv_file = pd.read_csv(csv_path, delimiter=',', 
	    names=['Absolute path', 'Relative path', 'Class'])  
        abs_path = csv_file['Absolute path']
        classes=csv_file['Class']
        for path_of_image in abs_path:
            img=cv2.imread(path_of_image)
            cv_tuple= img.shape
            height_image.append(cv_tuple[0])
            width_image.append(cv_tuple[1])
            channels_image.append(cv_tuple[2])
            if csv_file.loc[counter, 'Class'] == class_one:
                label=0
            else:
                label=1
            numerical.append(label)
            counter+=1
        csv_file=csv_file.drop('Relative path', axis=1)
        csv_file['Height']= height_image
        csv_file['Width']= width_image
        csv_file['Channels']= channels_image
        csv_file['Label']= numerical
        csv_file.to_csv(new_csv_filename, index=False)
        print(csv_file)

if __name__ == "__main__":
    with open('Lab4\settings.json', "r") as settings_file:
       settings = json.load(settings_file)

    image_forms("Lab2\csv_files\dataset.csv", 'Lab4\data.csv',settings["class_one"])
