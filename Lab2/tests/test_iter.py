import argparse
import unittest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))
from src.write_reader import *       


class TestWriteReaderIter(unittest.TestCase):

    def setUp(self):
        with open(os.path.join('Lab2/tests/test_data.json'), 'r') as f:          
            test_data = json.load(f)  
            print(len(test_data))                  
        self.dataset = DataWriteReader('test_data.json',*test_data)        
        
   
    def test_next_class(self):                
        self.assertEqual(self.dataset.next("class1") , "rel_class11")
        self.assertEqual(self.dataset.next("class2") , "rel_class12")
        self.assertEqual(self.dataset.next("class1") , "rel_class21")
        self.assertEqual(self.dataset.next("class3") , "rel_class13")
         
    def test_iter_for_class(self):
        items1 : list[str] = list()
        items2 : list[str] = list()
        items3 : list[str] = list()
        for item in self.dataset.class_list("class1"):
            items1.append(item)        

        for item in self.dataset.class_list("class2"):
            items2.append(item)
        for item in self.dataset.class_list("class3"):
            items3.append(item)    
        self.assertEqual(len(items1), 5)    
        self.assertEqual(len(items2), 3)
        self.assertEqual(len(items3), 2)

if __name__ == '__main__':
    unittest.main()
    
        