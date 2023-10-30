import os
import shutil
import json
import csv
import random
import logging
from typing import Optional


class DataWriteReader:
    '''Class for working with file data\n
       and designed to store file data in the field of objects\n
       of this class in the form of dictionaries.
       Manipulate this data:\n
       reading-writing data to CSV;\n
       copying files to another directory in various formats.
    '''

    key1 : str = 'absolute path'
    key2 : str = 'relative path'
    key3 : str = 'class label'
    
    def create_dir(dir:str) -> None:
        '''This static method that creates a directory for the dataset at the specified address'''
        try:
            os.makedirs(os.path.join(dir))
        except FileExistsError as e:
            logging.error(f'[{e}] A folder with the same name already exists') 
    
    def __init__(self, main_dir: str, *args : Optional[dict[str,str]]):
        '''main_dir: directory from which files-data will be extracted\n
           args: files-data set (optional)
        '''
        self.main_dir : str = main_dir        
        if args:
            self.data_list : list[dict[str, str]] = list(args)
        else:    
            self.data_list : list[dict[str, str]] = list()
            self.scan_instances(main_dir)        

    def scan_instances(self, main_dir : str) -> None:
        '''This method scans the directory and saves the data\n
           of all found files in the current object field
        '''
        try:
            catalog_list : list[str] = os.listdir(main_dir)
            for catalog in catalog_list:
                try:
                    file_list : list[str] = os.listdir(os.path.join(main_dir,catalog))
                    if file_list:
                        for file in file_list:
                            relative_path : str = os.path.join(main_dir, catalog, file)
                            absolute_path : str = os.path.abspath(relative_path)
                            class_label : str = catalog.replace('_',' ')
                            self.data_list.append({DataWriteReader.key1: absolute_path,
                                                   DataWriteReader.key2: relative_path,
                                                   DataWriteReader.key3: class_label})                            
                    else:
                        logging.warning(f"directory '{catalog}' is empty")   
                except NotADirectoryError:
                        relative_path : str = os.path.join(main_dir, catalog)
                        absolute_path : str = os.path.abspath(catalog)
                        class_label : str = catalog[:-4]
                        self.data_list.append({DataWriteReader.key1: absolute_path,
                                               DataWriteReader.key2: relative_path,
                                               DataWriteReader.key3: class_label})       
        except FileExistsError as e:             
            logging.error(f"[{e}] Directory '{main_dir}' does not exist")          

    def write_to_csv(self, path: str) -> None:
        '''This method writes file data contained in the current object to a CSV file'''   
        with open(path, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.data_list[0].keys())
            for dict_item in self.data_list:
                csv_writer.writerow(dict_item.values())

    def read_from_csv(self, path: str) -> list[dict[str, str]]:
        '''This method reads file data from CSV\n 
           and returns a new object of the current class with the read data
        '''
        try:
            with open(path, 'r', newline='') as csvfile:
                spamreader = csv.reader(csvfile, quotechar='|')
                data_list : list[dict[str, Optional[str]]] = list()
                for row in spamreader:     
                    data_list.append({DataWriteReader.key1: row[0],
                                      DataWriteReader.key2: row[1],
                                      DataWriteReader.key3: row[2]}) 
            return data_list       
        except FileNotFoundError as e:  
            logging.error(f"[{e}] CSV '{path}' does not exist") 

    def copy_dataset_num_class(self, path: str) -> "DataWriteReader": 
        '''This method copies files to the specified directory\n 
           and renames the original files in the format "serial number_classlabel.jpg"\n
           Returns a new object of the current class containing the data of the copied files
        '''           
        DataWriteReader.create_dir(path)
        number : int = lambda path: (path[:-4])[-4:]
        key1 : str = DataWriteReader.key1
        key2 : str = DataWriteReader.key2
        key3 : str = DataWriteReader.key3
        new_data : list[dict[str, str]] = list()
        for dict_item in self.data_list:
            new_path : str = os.path.join(path, f"{dict_item[key3]}_{number(dict_item[key2])}.jpg")
            new_dict : int = dict()
            shutil.copyfile(dict_item[key2], new_path)            
            new_dict[key1] = os.path.abspath(new_path)
            new_dict[key2] = os.path.join(new_path)
            new_dict[key3] = dict_item[key3]
            new_data.append(new_dict)
        return DataWriteReader(path, *new_data)  
      
    def copy_dataset_randnum(self, path: str) -> "DataWriteReader":
        '''This method copies files to the specified directory\n 
           and renames the original files in the format "rand_num.jpg"\n
           Returns a new object of the current class containing the data of the copied files
        '''
        DataWriteReader.create_dir(path)
        rand_nums = list(range(1,10000))
        random.shuffle(rand_nums)
        key1 : str = DataWriteReader.key1
        key2 : str = DataWriteReader.key2
        key3 : str = DataWriteReader.key3
        new_data : list[dict[str, str]] = list()
        i : int = 0
        for dict_item in self.data_list:
            new_dict : dict = dict()
            new_path : str = os.path.join(path, f"{rand_nums[i]}.jpg")
            shutil.copyfile(dict_item[key2], new_path)
            new_dict[key1] = os.path.abspath(new_path)
            new_dict[key2] = os.path.join(new_path)
            new_dict[key3] = dict_item[key3]
            new_data.append(new_dict)
            new_data = sorted(new_data, key=lambda x: x[key2])
            i += 1
        return DataWriteReader(path, *new_data)   