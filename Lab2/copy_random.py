import numbers
import os
import logging
import argparse
import random
import create_annotation
import shutil


logging.basicConfig(level=logging.INFO)


def get_random_list() -> list:
    rand_list = list(i for i in range(0, 10000))
    random.shuffle(rand_list)
    return rand_list


def copy_random(old_dir: str, classes: str, new_dir: str, csv:str) -> None:
    try:
        csv_list = list()
        rand_list = get_random_list()
        j = 0
        for c in classes:
            count = len(os.listdir(os.path.join(old_dir, c)))
            for i in range(count):
                old = os.path.abspath(os.path.join(old_dir, c, f"{i:04}.txt"))
                new = os.path.abspath(os.path.join(new_dir, f"{rand_list[j]:04}.txt"))
                shutil.copy(old, new)
                row = [[new, os.path.relpath(new), c]]
                csv_list += row
                j += 1
        create_annotation.write_into_csv(csv, csv_list)
    except Exception as exc:
        logging.error(f"Can not write: {exc.message}\n{exc.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file name for annotation file, path of dataset')
    parser.add_argument('--old_path', type=str, default='dataset', help='Input path of dataset')
    parser.add_argument('--new_path', type=str, default='rand_dataset', help='Input path of new dataset')
    parser.add_argument('--classes', type=list, default=['1', '2', '3', '4', '5'])
    parser.add_argument('--name', type=str, default='annotation_rand', help='Input name for annotation')
    args = parser.parse_args()
    copy_random(args.old_path, args.classes, args.new_path, args.name)


