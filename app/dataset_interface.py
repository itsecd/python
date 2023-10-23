import os
import csv
import logging

class DataAbstract:

    def make_new_fold(name_csv : str, img_classes : list, directory : str) -> None:
        pass

    
    def create_csv(name_csv : str) -> None:
        """
        Create csv file

        Create csv file using a name_csv
        Parameters
        ----------
        name_csv : str
            Name of csv file
        """
        try:
            if not os.path.exists(f"{name_csv}.csv"):
                with open(f"{name_csv}.csv", "w", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(("Absolute path", "Relative path", "Class"))
        except Exception as ex:
            logging.error(f"Error create csv file: {ex}")


    def write_in_csv(name_csv : str, img_class : str, directory : str) -> None:
        """
        Write in csv file

        Wrute in csv file
        Parameters
        ----------
        name_csv : str
            Name of csv file
        img_class : str
            Name of object
        directory : str
            Directory where is our img
        """
        try:    
            row = [
                os.path.abspath(directory),
                directory,
                img_class
            ]
            with open(f"{name_csv}.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except Exception as ex:
            logging.error(f"Error of writing row in csv: {ex}")

