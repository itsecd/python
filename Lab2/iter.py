from path_of_next import get_next


class DirectoryIterator:
    def __init__(self, class_name: str, csv_path: str) -> None:
        self._counter = 0
        self._class_name = class_name
        self._csv_path = csv_path
        self._paths = get_next(class_name, csv_path)
        self._limit = len(self._paths) if self._paths else 0

    @property
    def class_name(self) -> str:
        return self._class_name

    @property
    def paths(self) -> list[str]:
        return self._paths

    @property
    def limit(self) -> int:
        return self._limit

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter < self._limit:
            next_path = self._paths[self._counter]
            self._counter += 1
            return next_path
        else:
            raise StopIteration


if __name__ == "__main__":
    csv_path = "annotation.csv"

    cat = DirectoryIterator('cat', csv_path)
    dog = DirectoryIterator('dog', csv_path)
    
    print(cat.class_name)
    print(cat.paths)
    print(cat.limit)

    print(next(cat))
    print(next(cat))
    print(next(cat))

    print(dog.class_name)
    print(dog.paths)
    print(dog.limit)

    print(next(dog))
    print(next(dog))
    print(next(dog))
