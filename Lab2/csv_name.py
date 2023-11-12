import csv
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

def make_csv(eggs: str) -> None:
    '''принимает имя папки и создает csv файл'''
    try:
        if not os.path.exists(eggs):
            # возвращает, True если path ссылается на существующий путь или открытый файловый дескриптор
            with open(f'{eggs}.csv', 'a') as csvfile: 
                csv.writer(csvfile, lineterminator="\n") # Возвращает объект writer, ответственный за преобразование 
                # пользовательских данных в строки с разделителями для данного файлоподобного объекта
    except:
        logging.error(f'Error in make_csv')

def make_list(
    folder: str, 
    search: str
) -> list:
    '''принимает название папки и метки класса (cat, dog) '''
    list = []
    for s in search:
        for photo in range(main["max_file"]):
            string = [ [ # возвращает нормализованную абсолютизированную версию имени пути path
                os.path.abspath(os.path.join(folder, s, f"{photo:04}.jpg")), # абсолютный путь
                os.path.join(folder, s, f"{photo:04}.jpg"), # dataset\cat\0001.jpg
                s, ] # cat
            ]
            list += string
    return list

def write_in_file(
    search: list, 
    eggs: str
) -> None:
    '''принимает название строк для таблицы 
    и имя папки, в которую пойдет запись'''
    try:
        make_csv(eggs)
        for s in search:
            with open(f'{eggs}.csv', 'a', newline='') as csvfile:
                # добавление; указатель на конце файла, если файл есть; создает новый файл, если его нет
                spamwriter = csv.writer(csvfile, lineterminator="\n")
                spamwriter.writerow(s)
    except:
        logging.error(f'Error in write_in_file')

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        # чтение; указатель на начале файла; вызывается по умолчанию
        main = json.load(main_file)

    write_in_file(make_list("dataset", main["search"]), "Lab2\set\dataset_another")
    
