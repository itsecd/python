import csv
from dataset_manager import get_next_instance


class ClassIterator:
    def __init__(self, csv_file: str, class_labels: list):
        """Initialize a ClassIterator object.
        Parameters:
        - csv_file (str): The path to the CSV file.
        - class_labels (str): List of class labels.
        """
        self.csv_file = csv_file
        self.class_labels = class_labels
        self.current_label_index = 0

    def get_next_instance(csv_file: str, current_label: str) -> str:
        return get_next_instance(csv_file, current_label)

    def get_next_instance_for_current_label(self) -> str:
        """
        Get the next instance for the current class label.
        Returns:
        Optional[str]: The path to the next instance of the current class label, or None if instances are exhausted.
        """
        if not self.class_labels or self.current_label_index >= len(self.class_labels):
            return None

        current_label = self.class_labels[self.current_label_index]
        next_instance = self.get_next_instance(self.csv_file, current_label)

        if next_instance is not None:
            self.current_label_index += 1
            return next_instance
        else:
            return None


if __name__ == "__main__":
    iterator = ClassIterator("copy_dataset.csv", ["tiger", "leopard"])
