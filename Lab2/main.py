import os
import json
import csv_annotation
import new_name_copy
import random_of_copy

if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)


print("Введите режим:")
print("normal-составление файла аннотации ")
print("new_name-копирование и изменение имени файлов(класс_номер)")
print("random-копирование и изменение имени файлов(рандоманое число от 0 до 10 000)")
mode = input()


if mode == "normal":
    l = csv_annotation.make_list(fcc["main_folder"], fcc["classes"])
    csv_annotation.write_in_file(
        (os.path.join(settings["directory"],
         settings["folder"], settings["normal"])), l
    )

if mode == "new_name":
    new_name_copy.copy_in_new_directory(
        fcc["main_folder"],
        fcc["classes"],
        fcc["main_folder"],
        (os.path.join(settings["directory"],
         settings["folder"], settings["new_name"])),
    )

if mode == "random":
    random_of_copy.copy_with_random(
        fcc["main_folder"],
        fcc["classes"],
        fcc["main_folder"],
        (os.path.join(settings["directory"],
         settings["folder"], settings["random"])),
    )
