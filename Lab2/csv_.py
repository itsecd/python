import os
import csv
from dotenv import load_dotenv
from pathlib import Path
import logging

os.chdir("C:\\Users\\Yana\\Documents\\python-v6\\")
logging.basicConfig(filename="Lab2\\py_log1.log", filemode="a", level=logging.INFO)

def make_csv(name_csv:str) -> None:
    """This function creates a csv
    file with specific name"""
    logging.info("in function make_csv")
    try:
        if not os.path.exists(name_csv):
            with open(f"Lab2\\{name_csv}.csv","w") as f:
                csv.writer(f)
    except Exception as e:
        logging.error(f"file not created:{e}")

def make_list(directory:str,tags:list) -> list:
    """Function creates list with 
    parametrs for csv file"""
    img_list=[]
    for tag in tags:
        logging.info(f"for {tag}")
        count_files=len(os.listdir(os.path.join(directory,tag)))
        for img in range(count_files):
            item=[[os.path.abspath(os.path.join(directory,tag,f"Lab2\\{img:04}.jpg")), os.path.join(directory,tag,f"{img:04}.jpg"),tag]]
            img_list+=item
    return img_list

def write_scv(name_csv:str, img_list:list) -> None:
    """Function get data for each img in list 
    and writes them to csv file"""
    try:
        make_csv(name_csv)
        logging.info("make_csv")
        for img in img_list:
            with open(f"{name_csv}.csv","a") as f:
                csv.writer(f,lineterminator="\n").writerow(img)
        logging.info('csv done!')
    except Exception as e:
        logging.error(f"Error in write data:{e}")

if __name__=="__main__":
    load_dotenv()
    env_path = Path('C:\\Users\\Yana\\Documents\\python-v6\\Lab1')/'.env'
    load_dotenv(dotenv_path=env_path)
    tiger=os.getenv("FOLDER_TIGER")
    leopard=os.getenv("FOLDER_LEOPARD")
    l=[tiger,leopard]
    logging.info("start")
    list=make_list(os.path.join("Lab1","dataset"),l)
    write_scv("Lab2\\file",list)