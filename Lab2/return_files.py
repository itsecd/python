import csv


def get_paths(class_name: str, csv_path: str) -> list:
    """
    This function returns the relative paths for the class objects passed
    to the function using information from a CSV file.
    """
    paths = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for row in reader:
            if class_name in row[2]:
                paths.append(row[1])

    return paths if paths else None 