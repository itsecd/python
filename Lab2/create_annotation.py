import csv
import os
import logging
import argparse


logging.basicConfig(level=logging.INFO)


def create_csv(name: str) -> None:
    try: 
        if not os.path.exists(name):
            with open(f"{name}.csv", "a") as file:
                csv.writer(file, lineterminator="\n")
    except Exception as exc:
        logging.error(f"Can not create file: {exc.message}\n{exc.args}\n")


def create_csv_list(directory: str, classes: str) -> list:
    csv_list = list()
    for c in classes:
        count = len(os.listdir(os.path.join(directory, c)))
        for i in range(count):
            row = [[os.path.abspath(os.path.join(directory, c, f"{i:04}.txt")), os.path.join(directory, c, f"{i:04}.txt"), c]]
            csv_list += row
    return csv_list


def write_into_csv(name: str, csv_list: list) -> None:
    try:
        create_csv(name)
        for c in csv_list:
            with open(f"{name}.csv", "a") as file:
                write = csv.writer(file, lineterminator="\n")
                write.writerow(c)
    except Exception as exc: 
        logging.error(f"Can not save/write data: {exc.message}\n{exc.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file name for annotation file, path of dataset')
    parser.add_argument('--path', type=str, default='dataset', help='Input path of dataset')
    parser.add_argument('--name', type=str, default='annotation_og', help='Input name for annotation')
    parser.add_argument('--classes', type=list, default=['1', '2', '3', '4', '5'])
    args = parser.parse_args()
    csv_list = create_csv_list(args.path, args.classes)
    write_into_csv(args.name, csv_list)