import csv
import os
import json

def choice_path(
    eggs: str, 
    choice: str,
    search: str
) -> None:
    '''принимает название папки, из которой считывем; 
    имя папки, в которую записываем и метку класса (cat, dog)'''
    ch = []
    with open(f'{eggs}.csv', 'r') as r_file:
        file_reader = csv.reader(r_file, delimiter = ",")
        for a in file_reader:
            ch.append(a)
        if not os.path.exists(eggs):
                with open(f'{choice}.csv', 'w') as csvfile: 
                    spamwriter = csv.writer(csvfile, lineterminator="\n")
                    spamwriter.writerow(["Path"])
                    for i in range(main["max_file"]):
                        if ch[i + 1][2] == search:
                            spamwriter.writerow({ch[i + 1][1]})
                        elif ch[i + 1 + main['max_file']][2] == search:
                            spamwriter.writerow({ch[i + 1 + main['max_file']][1]})
                        else:
                            return None

if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        main = json.load(main_file)

    choice_path(main["folder_an"], main["folder_choice"], "cat")
