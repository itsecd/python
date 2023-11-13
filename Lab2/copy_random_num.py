import os
import json
import shutil
import logging
import random
import csv_

logging.basicConfig(filename="Lab2\\py_log1.log", filemode="a", level=logging.INFO)

def copy_with_random_num(old_dir:str, tags:str, new_dir: str, name_csv:str) -> None:
    """Function for copying images from tags folders to the dataset
    with with a name - random number"""
    try:
        img_list=[]
        random_num=set()
        count_files=len(os.listdir(os.path.join(old_dir,tags[0])))
        while len(random_num) <= (count_files*len(tags)):
            random_num.add(random.randint(0,1000))
        numbers=list(random_num)
        for tag in tags:
            for i in range(count_files):
                j=len(os.listdir(os.path.join(new_dir)))-len(tags)
                b=os.path.abspath(os.path.join(old_dir,tag,f"{i:04}.jpg"))
                a=os.path.abspath(os.path.join(new_dir,f"{numbers[j]:04}.jpg"))
                shutil.copy(b,a)
                img=[[a,os.path.relpath(a),tag]]
                img_list+=img
        csv_.write_scv(name_csv,img_list)
        logging.info("copy with random num done")
    except Exception as e:
        logging.error(f"copy_with_random_num error{e}")

if __name__=="__main__":
    with open(os.path.join("Lab2","settings.json"),'r') as f:
        settings=json.load(f)
    copy_with_random_num(settings['main_folder'],settings["tags"],"Lab2\\dataset","Lab2\\copy_file_random")