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
            with open(f'{eggs}.csv', 'w', newline='') as csvfile: # запись; перезаписывает файл, если он есть; создает новый файл, если его нет
                spamwriter = csv.writer(csvfile) # Возвращает объект writer, ответственный за преобразование 
                # пользовательских данных в строки с разделителями для данного файлоподобного объекта
                spamwriter.writerow(['Absolute path', 'Relative path', 'Class'])
    except:
        logging.error(f'Error in make_csv')


def write_in_file(
    search: str, 
    folder: str, 
    eggs: str
) -> None:
    '''принимает название папки, подназвание (cat, dog) 
    и имя папки, в которую пойдет запись'''
    try:
        make_csv(eggs)
        for s in search:
            counts = len(os.listdir(os.path.join(folder, s)))
            # возвращает список всех файлов и каталогов в данном каталоге
            for photo in range(counts):
                string = [ # возвращает нормализованную абсолютизированную версию имени пути path
                    os.path.abspath(os.path.join(s, f"{photo:04}.jpg")), # (cat\0001.jpg)
                    os.path.join(folder, s, f"{photo:04}.jpg"), # dataset\cat\0001.jpg
                    s, # cat
                ]
                with open(f'{eggs}.csv', 'a', newline='') as csvfile:
                    # добавление; указатель на конце файла, если файл есть; создает новый файл, если его нет
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(string)
    except:
        logging.error(f'Error in write_in_file')

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        # чтение; указатель на начале файла; вызывается по умолчанию
        main = json.load(main_file)

    write_in_file(main["search"], main["folder"], "dataset_another")