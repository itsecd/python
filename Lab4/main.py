# main.py
import pandas as pd
from functions import add_pixel_count_column, group_by_class_and_compute_stats
from histogram import plot_histogram, plot_histograms

# Загрузка данных
data = [
    {'Название класса': 'Класс1', 'Абсолютный путь к файлу': 'C:\Users\Prodigy-\Desktop\123\222.csv'},
    {'Название класса': 'Класс2', 'Абсолютный путь к файлу': 'C:\Users\Prodigy-\Desktop\123\222.csv'},
    # Добавьте другие элементы по мере необходимости
]
df = pd.DataFrame(data)

# Добавление столбца с количеством пикселей
df = add_pixel_count_column(df)

# Группировка по классу и вычисление статистики
grouped_stats = group_by_class_and_compute_stats(df)
print("Статистика по количеству пикселей:")
print(grouped_stats)

# Выбор случайного класса для построения гистограммы
random_class = df['название_класса'].sample(1).values[0]
print(f"\nСтроим гистограмму для класса: {random_class}")

# Построение гистограммы
hist_b, hist_g, hist_r = plot_histogram(df, random_class)

# Отрисовка гистограмм
plot_histograms(hist_b, hist_g, hist_r)
