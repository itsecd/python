import csv
import os
#"C:\Users\korsy\OneDrive\Рабочий стол\pythonlabs\Lab2\dataset\1\0001.txt"
#dataset\1\0001.txt
#1
rewievs=[[]]
for star in range(1,6):
    dir='dataset'
    directory = os.path.join(dir, f"{star}")
    files = [file for file in os.listdir(directory) if os.path.isfile(f'{directory}/{file}')]
    cnt_files=len(files)
    for file in range(1,cnt_files+1):
        relative_path=os.path.join(directory, f"{str(file).zfill(4)}.txt")
        absolute_path = os.path.abspath(relative_path)
        rewievs.append([relative_path,absolute_path,star])
filename='rewievs'
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rewievs)

