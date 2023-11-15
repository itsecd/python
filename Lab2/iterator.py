import os


class ElementIterator:
    def __init__(self, class_name:str) -> None:
        self.counter = 0
        self.class_name = class_name
        self.data = os.listdir(os.path.join('dataset', self.class_name))
        self.limit = len(self.data)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.counter < self.limit:
            next_path = os.path.join(self.class_name, self.data[self.counter])
            self.counter += 1
            return next_path
        else:
            return None
    

if __name__ == "__main__":

    polarbears = ElementIterator('polar bear')
    brownbears = ElementIterator('brown bear')

    print(next(polarbears))
    print(next(polarbears))
    print(next(polarbears))
    print(next(brownbears))
    print(next(brownbears))
    print(next(brownbears))