import os
import json
import logging
import random
import csv_name

logging.basicConfig(level=logging.INFO)


def write_in_new(
    folder: str,
    search: str,
    name_csv: str,
    choice: int
) -> None:
    '''принимает название папки, метку класса (cat, dog), имя папки, в которую пойдет запись и выбор записи'''
    try:
        string = []
        for s in search:
            for photo in range(main["max_file"]):
                if choice == 0:
                    new = os.path.abspath(os.path.join(
                        folder, f"{s}_{photo:04}.jpg"))
                    a = [[new, os.path.relpath(new), s]]
                    string += a
                else:
                    b = random.randint(0, 10000)
                    new = os.path.abspath(os.path.join(folder, f"{b:04}.jpg"))
                    a = [[new, os.path.relpath(new), s]]
                    string += a
        csv_name.write_in_file(string, name_csv)
    except:
        logging.error(f"Error in write_in_file")


if __name__ == "__main__":
    with open(os.path.join("Lab2", "main.json"), "r") as main_file:
        main = json.load(main_file)

    write_in_new(main["folder"], main["search"], main["folder_new"], 0)
    write_in_new(main["folder"], main["search"], main["folder_random"], 1)
