import os
import logging
import shutil
from dataset_interface import *
from get_way import get_path_normal, get_path_together


class MakeTogetherData(DataAbstract):

    def make_new_fold(name_csv : str, img_classes : list, directory : str) -> None:
        """
        Make new csv file

        Make new csv file
        Parameters
        ----------
        name_csv : str
            Name of csv file
        img_classes : str
            List of objects
        directory : str
            Path to dataset
        """
        try:
            DataAbstract.create_csv(name_csv)
            for img_class in img_classes:
                number_of_img = len(os.listdir(os.path.join(directory, img_class)))
                for img in range(number_of_img):
                    shutil.copyfile(get_path_normal(img_class, img), get_path_together(img_class, img))
                    DataAbstract.write_in_csv(name_csv, img_class, get_path_together(img_class, img))
        except Exception as ex:
            logging.error(f"Error of copy img : {img} | {ex}")


    # if __name__ == "__main__":
    #     with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
    #         fj = json.load(fjson)

    #     with open(os.path.join("Lab2", "src_csv.json"), 'r') as srcjson:
    #         sj = json.load(srcjson)

    #     if not os.path.isdir(os.path.join(fj["main_folder"], sj["together"])):
    #         os.mkdir(os.path.join(fj["main_folder"], sj["together"]))

    #     logging.basicConfig(level=logging.INFO)
    #     make_new_fold(os.path.join("Lab2", sj["together"]), fj["objects"], fj["main_folder"])