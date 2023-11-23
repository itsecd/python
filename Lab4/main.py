import pandas as pd
import logging
import cv2
import json

logging.basicConfig(level=logging.INFO)

def image_forms(csv_path:str, new_csv_filename:str, class_one) -> pd.DataFrame:
        height_image = []
        width_image = []
        channels_image = []
        numerical = []
        counter = 0  # счетчик
        csv_file = pd.read_csv(csv_path, delimiter=',', 
	    names=['Absolute path', 'Relative path', 'Class'])  
        abs_path = csv_file['Absolute path']
        
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
        return csv_file

def checking_balance(dataframe_file:str, statistic_file:str) -> pd.DataFrame:
    csv_file = pd.read_csv(dataframe_file)
    images_info=csv_file[['Height', 'Width','Channels']].describe()
    label_stats=csv_file['Label']
    label_info=label_stats.value_counts()
    df = pd.DataFrame() 
    num_images_per_label = label_info.values
    is_balanced = (num_images_per_label[0]/num_images_per_label[1])
    df['Quantity']=num_images_per_label
    df['Balance']= f'{is_balanced:.1f}'
    pd.concat([images_info,df], axis=1).to_csv(statistic_file)
    if ( is_balanced>=0.95 and is_balanced<=1.05):
        print("Выборка сбалансированна")
    else:
        print("Выборка несбалансирована с точностью:", f"{abs(is_balanced*100-100):.1f}%")
    return df
   
def filter_by_label(dframe, label) -> pd.DataFrame:
    filtered_df = dframe[dframe['Label'] == label]
    return filtered_df

def min_max_filter(dframe, width_max: int, height_max: int, label: str) -> pd.DataFrame:
    filtered_df = dframe[(dframe['Label'] == label) & (dframe['Width'] <= width_max) & (dframe['Height'] <= height_max)]
    return filtered_df

if __name__ == "__main__":
    with open('Lab4\settings.json', "r") as settings_file:
       settings = json.load(settings_file)

    #image_forms("Lab2\csv_files\dataset.csv", 'Lab4\data.csv',settings["class_one"])
    #checking_balance("Lab4\data.csv", "Lab4\my_stat.csv")
    csv_file = pd.read_csv('Lab4\data.csv')
    new=min_max_filter(csv_file, 200, 200, 1)
    print(new)
