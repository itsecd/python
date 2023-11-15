import csv
import os
import json


def choice_path(
    name_csv: str,
    choice: str,
    search: str
) -> None:
    '''принимает название папки, из которой считывем; 
    имя папки, в которую записываем и метку класса (cat, dog)'''
    ch = []
    with open(f'{name_csv}.csv', 'r') as r_file:
        file_reader = csv.reader(r_file, delimiter=",")
        for a in file_reader:
            ch.append(a)
        if not os.path.exists(name_csv):
            with open(f'{choice}.csv', 'w') as csvfile:
                spamwriter = csv.writer(csvfile, lineterminator="\n")
                for i in range(main["max_file"]):
                    if ch[i][2] == search:
                        spamwriter.writerow({ch[i][1]})
                    elif ch[i + main['max_file']][2] == search:
                        spamwriter.writerow({ch[i + main['max_file']][1]})
                    else:
                        return None


if __name__ == "__main__":
    with open(os.path.join("Lab2", "main.json"), "r") as main_file:
        main = json.load(main_file)

    choice_path(main["folder_an"], main["folder_choice"], "cat")
