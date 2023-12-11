import pandas as pd
import matplotlib.pyplot as plt
from csv_file_processing import preparing_dataframe, filter_by_date_range, filter_by_mean_deviation, group_by_month
from creating_graphs import create_graph, create_monthly_graph

file_path = "Lab4/dataset/data.csv"

if __name__ == "__main__":
    resulting_dataframe = preparing_dataframe(file_path)
    print("Выберите действие:")
    print("1. Фильтрация по отклонению от среднего значения курса")
    print("2. Фильтрация по дате")
    print("3. Фильтрация по месяцу")
    print("4. Построение графика для всего периода")
    print("5. Построение графика для определенного месяца")

    choice = input("Введите номер действия: ")

    if choice == "1":
        deviation_value = float(input("Введите значение отклонения от среднего значения курса: "))
        filtered_data = filter_by_mean_deviation(resulting_dataframe, deviation_value)
        print(filtered_data)
    elif choice == "2":
        start_date = input("Введите начальную дату в формате YYYY-MM-DD: ")
        end_date = input("Введите конечную дату в формате YYYY-MM-DD: ")
        filtered_data = filter_by_date_range(resulting_dataframe, start_date, end_date)
        print(filtered_data)
    elif choice == "3":
        filtered_data = group_by_month(resulting_dataframe)
        print(filtered_data)
    elif choice == "4":
        create_graph(file_path)
    elif choice == "5":
        month_to_plot = input("Введите номер месяца для построения графика(YYYY-MM): ")
        filtered_data = group_by_month(resulting_dataframe)
        create_monthly_graph(filtered_data, month_to_plot)
    else:
        print("Выбрано недопустимое действие. Пожалуйста, выберите номер действия от 1 до 3.")