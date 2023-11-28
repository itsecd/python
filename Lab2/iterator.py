from return_files import get_paths


class DatasetIterator:
    def __init__(self, class_name: str, 
                 csv_path: str) -> None:
        self.counter = 0
        self.class_name = class_name
        self.paths = get_paths(class_name, csv_path)
        self.limit = len(self.paths) if self.paths else 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.counter < self.limit:
            next_path = self.paths[self.counter]
            self.counter += 1
            return next_path
        else:
            return None