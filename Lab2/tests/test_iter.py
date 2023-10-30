import argparse
import unittest
import sys
import os
import json
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.class_iter import ClassInstanceIterator 

     
class TestWriteReaderIter(unittest.TestCase):

    def setUp(self):
        with open(os.path.join('Lab2/tests/test_data.json'), 'r') as f:          
            self.test_data = json.load(f)       
   
    def test_next_class(self):
        test_iter = ClassInstanceIterator(self.test_data)

        test_iter.change_class_label("class1")                
        self.assertEqual(next(test_iter) , "rel_class11")

        test_iter.change_class_label("class2")
        self.assertEqual(next(test_iter) , "rel_class12")

        test_iter.change_class_label("class1")
        self.assertEqual(next(test_iter) , "rel_class21")

        test_iter.change_class_label("class3")
        self.assertEqual(next(test_iter) , "rel_class13")
         
    def test_iter_for_class(self):
        items1 : list[str] = list()
        items2 : list[str] = list()
        items3 : list[str] = list()
        test_iter = ClassInstanceIterator(self.test_data)
        test_iter.change_class_label("class1")
        for item in test_iter:
            items1.append(item)        

        test_iter.change_class_label("class2")
        for item in test_iter:
            items2.append(item)

        test_iter.change_class_label("class3")    
        for item in test_iter:
            items3.append(item)    
        self.assertEqual(len(items1), 5)    
        self.assertEqual(len(items2), 3)
        self.assertEqual(len(items3), 2)

if __name__ == '__main__':
    unittest.main()
    
        