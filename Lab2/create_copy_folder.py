import os
import logging
import argparse
import create_annotation
import shutil

logging.basicConfig(level=logging.INFO)


def copy_in_dir(old_dir: str, classes: str, new_dir: str, csv:str) -> None:
    try:
        csv_list = list()
        for c in classes:
            count = len(os.listdir(os.path.join(old_dir, c)))
            for i in range(count):
                old = os.path.abspath(os.path.join(old_dir, c, f"{i:04}.txt"))
                new = os.path.abspath(os.path.join(new_dir, f"{c}_{i:04}.txt"))
                shutil.copy(old, new)
                row = [[new, os.path.relpath(new), c]]
                csv_list += row
        create_annotation.write_into_csv(csv, csv_list)
    except Exception as exc:
        logging.error(f"Can not write: {exc.message}\n{exc.args}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file name for annotation file, path of dataset')
    parser.add_argument('--old_path', type=str, default='dataset', help='Input path of dataset')
    parser.add_argument('--new_path', type=str, default='new_dataset', help='Input path of new dataset')
    parser.add_argument('--classes', type=list, default=['1', '2', '3', '4', '5'])
    parser.add_argument('--name', type=str, default='annotation_copy', help='Input name for annotation')
    args = parser.parse_args()
    copy_in_dir(args.old_path, args.classes, args.new_path, args.name)