import csv


def get_path(name_csv: str, name_class: str) -> None:
    data = list()
    with open(name_csv, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    limit = len(data)
    for i in range(limit):
        if data[i][2] == name_class:
            count = i - 1
            break
    name_class
    if count < limit:
        if name_class == data[count + 1][2]:
            count += 1
        return data[count][0]
    else:
        return None


if __name__ == "__main__":

    print(get_path("Lab2\csv_files\dataset_new.csv", "rose"))
