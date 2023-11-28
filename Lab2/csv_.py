import os
import csv
from dotenv import load_dotenv
from pathlib import Path
import logging

os.chdir("C:\\Users\\Yana\\Documents\\python-v6\\")
logging.basicConfig(filename="Lab2\\py_log1.log", filemode="a", level=logging.INFO)


def make_list(directory: str) -> list:
    """Function creates list with
    parametrs for csv file"""
    img_list = []
    count_files = len(os.listdir(os.path.join(directory)))
    logging.info(os.listdir(os.path.join(directory)))
    for item in os.listdir(os.path.join(directory)):
            str = [[os.path.abspath(os.path.join(directory,f"{item}")),
                    os.path.relpath(os.path.join(directory,f"{item}")),]]
            img_list += str
    return img_list


def write_csv(name_csv: str, img_list: list) -> None:
    """Function get data for each img in list
    and writes them to csv file"""
    try:
        logging.info("in function make_csv")
        logging.info("make_csv")
        for img in img_list:
            with open(f"Lab3\{name_csv}", "a") as f:
                csv.writer(f, lineterminator="\n").writerow(img)
        logging.info("csv done!")
    except Exception as e:
        logging.error(f"Error in write data:{e}")


if __name__ == "__main__":
    load_dotenv()
    env_path = Path("C:\\Users\\Yana\\Documents\\python-v6\\Lab1") / ".env"
    load_dotenv(dotenv_path=env_path)
    tiger = os.getenv("FOLDER_TIGER")
    leopard = os.getenv("FOLDER_LEOPARD")
    l = [tiger, leopard]
    logging.info("start")
    list = make_list(os.path.join("Lab1", "dataset"), l)
    write_csv("Lab2\\file", list)