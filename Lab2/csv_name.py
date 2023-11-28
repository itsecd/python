import csv
import os
import json
import logging

logging.basicConfig(level=logging.INFO)


def make_list(
    folder: str,
    search: str
) -> list:
    '''принимает название папки и метки класса (cat, dog)'''
    list = []
    for s in search:
        count = len(os.listdir(os.path.join(folder, s)))
        for photo in range(count):
            string = [[
                os.path.abspath(os.path.join(folder, s, os.listdir(os.path.join(folder, s))[photo])),
                os.path.relpath(os.path.abspath(
                    os.path.join(folder, s, os.listdir(os.path.join(folder, s))[photo]))),
                s, ]
            ]
            list += string
    return list


def write_in_file(
    string: list,
    name_csv: str
) -> None:
    '''принимает строки для таблицы 
    и имя папки, в которую пойдет запись'''
    try:
        if not os.path.exists(name_csv):
            for s in string:
                with open(f'{name_csv}.csv', 'a', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(s)
    except:
        logging.error(f'Error in write_in_file')


if __name__ == "__main__":
    with open(os.path.join("Lab2", "main.json"), "r") as main_file:
        main = json.load(main_file)

    write_in_file(make_list(main["folder"], main["search"]), main["folder_an"])
