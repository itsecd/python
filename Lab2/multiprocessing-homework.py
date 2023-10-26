import multiprocessing


def sum_row(i, matrix):
    if i <= len(matrix):
        sum = 0
        for w in range(len(matrix[i])):
            sum += matrix[i][w]
    return sum


if __name__ == '__main__':

    rows = int(input("Введите количество строк: "))
    cols = int(input("Введите количество столбцов: "))
    matrix = list()
    for i in range(rows):
        row = []
        for j in range(cols):
            element = int(input(f"Введите элемент [{i}][{j}]: "))
            row.append(element)
        matrix.append(row)
    index = []
    for i in range(rows):
        r = (i, matrix)
        index += tuple(r)
    total_sum = 0
    with multiprocessing.Pool(multiprocessing.cpu_count()*2) as p:
        r = p.starmap(sum_row, [(0, matrix), (1, matrix)])
    for j in range(len(r)):
        total_sum += r[j]
    print(total_sum)
    print(index)
