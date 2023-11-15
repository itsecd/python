import csv
from dataset_manager import get_next_instance


class ClassIterator:
    def __init__(self, csv_file: str, class_labels: list[str]):
        """Initializes a ClassIterator.
        - csv_file (str): The path to the CSV file.
        - class_labels (list[str]): List of class labels to iterate through.
        """
        self.csv_file = csv_file
        self.class_labels = class_labels
        self.current_label_index = 0

    def get_next_instance(self) -> str:
        """
        Retrieves the next instance for the current class label.

        Returns:
        - str: The path to the next instance, or None if instances are exhausted.
        """
        if not self.class_labels or self.current_label_index >= len(self.class_labels):
            return None

        current_label = self.class_labels[self.current_label_index]
        next_instance = get_next_instance(self.csv_file, current_label)

        if next_instance is not None:
            self.current_label_index += 1
            return next_instance
        else:
            return None


if __name__ == "__main__":
    iterator = ClassIterator("copy_dataset.csv", ["tiger", "leopard"])

    try:
        while True:
            next_instance = iterator.get_next_instance()
            if next_instance is not None:
                print(next_instance)
            else:
                print("No more instances.")
                break
    except Exception as e:
        print(f"An error occurred: {e}")
