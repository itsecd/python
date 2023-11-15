import csv

def get_next_instance(csv_file: str, class_label: str) -> str:
    data = []
    with open(csv_file, "r", newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    instances = [row[0] for row in data if row[2] == class_label]

    if instances:
        next_instance = instances.pop(0)
        new_data = [row for row in data if row[0] != next_instance]
        with open(csv_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_data)  
        return next_instance
    else:
        return None

# Пример использования
if __name__ == "__main__":
    print(get_next_instance("copy_dataset.csv", "tiger"))
