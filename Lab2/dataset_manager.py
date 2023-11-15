import csv
import shutil


def get_next_instance(csv_file: str, class_label: str) -> str:
    """Retrieves the next instance (path to it) of the specified class from the CSV file.
    csv_file (str): The path to the CSV file.
    class_label (str): The class label.
    Returns:
    str: The path to the next instance of the specified class, or None if instances are exhausted.
    """
    data = []
    with open(csv_file, "r", newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    instances = [row[0] for row in data if row[2] == class_label]

    if instances:
        next_instance = instances.pop(0)
        new_data = [row for row in data if row[0] != next_instance]

        temp_file = csv_file + '.tmp'
        with open(temp_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_data)

        shutil.move(temp_file, csv_file)

        return next_instance
    else:
        return None


if __name__ == "__main__":
    print(get_next_instance("copy_dataset.csv", "tiger"))
