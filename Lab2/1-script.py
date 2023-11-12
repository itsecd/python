import csv
import os


def get_annotation(dir='dataset') -> list[list[str, str, int]]:
    """
    the function creates list of lists consisting of three elements:
    relative path, absolute path and class label for each file.
    ----------
    dir : str
    """
    rewievs = [[]]
    for star in range(1, 6):
        directory = os.path.join(dir, f"{star}")
        files = [file for file in os.listdir(
            directory) if os.path.isfile(f'{directory}/{file}')]
        cnt_files = len(files)
        for file in range(1, cnt_files + 1):
            relative_path = os.path.join(
                directory, f"{str(file).zfill(4)}.txt")
            absolute_path = os.path.abspath(relative_path)
            rewievs.append([relative_path, absolute_path, star])
    return rewievs


def write_csv(path='rewievs') -> None:
    """
    the function writes list elements to a csv file.
    ----------
    path : str
    """
    rewievs = get_annotation()
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rewievs)


if __name__ == '__main__':
    write_csv()
