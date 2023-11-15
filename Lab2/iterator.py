from return_element import get_paths


class ElementIterator:
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
    

if __name__ == "__main__":

    csv_path = "paths.csv"

    polarbears = ElementIterator('polar bear', csv_path)
    brownbears = ElementIterator('brown bear', csv_path)
    print(next(polarbears))
    print(next(polarbears))
    print(next(polarbears))
    print(next(brownbears))
    print(next(brownbears))
    print(next(brownbears))