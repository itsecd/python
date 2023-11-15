import csv

def write_annotation_to_csv(file_path: str, data: list[list[str]]) -> None:
    """
    Write annotation data to a CSV file.
    """
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')
        writer.writerows(data)
