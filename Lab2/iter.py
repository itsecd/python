import os

class Iterator:
    def __init__(self, directory):
        self.counter = 0
        self.directory = directory
        self.data = os.listdir(os.path.join('dataset', directory))
        self.limit = len(self.data)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.counter < self.limit:
            next_path = os.path.join(self.directory, self.data[self.counter])
            self.counter += 1
            return next_path
        else:
            raise StopIteration

if __name__ == "__main__":
    cat_iterator = Iterator('cat')
    dog_iterator = Iterator('dog')

    for _ in range(4):
        print(next(cat_iterator))

    for _ in range(3):
        print(next(dog_iterator))