import os
import json
import shutil
import logging
import csv_

logging.basicConfig(filename="Lab2\\py_log1.log", filemode="a", level=logging.INFO)

def copy_dataset(old_dir:str, tags:str, new_dir:str, name_csv:str) -> None:
    """Function for copying images from tags folders to the dataset"""
    try:
        img_list=[]
        for tag in tags:
            count_files=len(os.listdir(os.path.join(old_dir,tag)))
            for i in range(count_files):
                b=os.path.abspath(os.path.join(old_dir,tag,f"{i:04}.jpg"))
                a=os.path.abspath(os.path.join(new_dir,f"{tag}_{i:04}.jpg"))
                shutil.copy(b,a)
                img=[[a,os.path.realpath(a),tag]]
                img_list+=img
        csv_.write_scv(name_csv,img_list)
        logging.info('copy file csv done')
    except Exception as e:
        logging.error(f"Write error(copy dataset) {e}")

if __name__=="__main__":
    with open(os.path.join("Lab2","settings.json"),'r') as f:
        settings=json.load(f)
    copy_dataset(settings['main_folder'],settings["tags"],"Lab2\\dataset","Lab2\\copy_file")