import os
import json
import random
import logging
import csv_name

logging.basicConfig(level=logging.INFO)

def write_in_random(
    folder: str, 
    search: list,  
    eggs: str
) -> None:
    '''принимает название папки, метку класса (cat, dog) и имя папки, в которую пойдет запись'''
    try:
        string = []
        for s in search:
            for photo in range(main["max_file"]):
                b = random.randint(0, 10000)
                new = os.path.abspath(os.path.join(folder, f"{b:04}.jpg"))
                a = [[new, os.path.relpath(new), s]]
                string += a
        csv_name.write_in_file(string, eggs)
    except:
        logging.error(f"Error in with_in_random")

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        main = json.load(main_file)

    write_in_random(main["folder"], main["search"], main["folder_random"])