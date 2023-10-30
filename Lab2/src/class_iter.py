from typing import Optional


class ClassInstanceIterator:
    '''The iterator class receives a class label as input 
       and returns the next instance (path to it) of this class
    '''

    def __init__(self, args : list[dict[str, str]], class_label : Optional[str] = None,):
        '''A constructor that receives path to CSV file with data,
           the key by which the class label will be compared
           and the class label by which the path to the next instance will be given;
           if the label is not specified,
           the iteration proceeds along the label of the first instance in the data set
        '''       
        self.__data: list[dict[str, str]] = list(args)
        self.__class_label : str
        self.__key = list(self.__data[0].keys())[2]
        if class_label:
            self.__class_label = class_label
        else:              
            self.__class_label = self.__data[0][self.__key]

    def __iter__(self) -> "ClassInstanceIterator":
        return self    

    def __next__(self) -> str:
        '''The method returns the first match of a class label in the data set, 
           and then removes it from the set
        '''
        index : int = 0
        while self.__data[index][self.__key] != self.__class_label:
            if index == (len(self.__data) - 1):
                raise StopIteration
            index += 1
        return (self.__data.pop(index))[self.__key]  
      
    def change_class_label(self, class_label : str) -> None:
        self.__class_label = class_label     