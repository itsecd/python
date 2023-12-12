import logging
import argparse
from csv_file_processing import preparing_dataframe, filter_by_date_range, filter_by_mean_deviation, group_by_month
from creating_graphs import create_graph, create_monthly_graph


logging.basicConfig(level=logging.INFO)


file_path = "Lab4/dataset/data.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get statistics on a csv file")
    resulting_dataframe = preparing_dataframe(file_path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--option1", action="store_true", help="Фильтрация по отклонению от среднего значения курса")
    group.add_argument("--option2", action="store_true", help="Фильтрация по дате")
    group.add_argument("--option3", action="store_true", help="Фильтрация по месяцу")
    group.add_argument("--option4", action="store_true", help="Построение графика для всего периода")
    group.add_argument("--option5", action="store_true", help="Построение графика для определенного месяца")

    parser.add_argument("--deviation_value",
                        type=float, default=30,
                        help="deviation value for filter")
    parser.add_argument("--start_date",
                        type=str, default="2015-10-10",
                        help="start date for filter")
    parser.add_argument("--end_date",
                        type=str, default="2016-10-10",
                        help="end date for filter")
    parser.add_argument("--month_to_plot",
                        type=str, default="2015-05",
                        help="month for filter")
    
    args = parser.parse_args()

    try:
        if args.option1:
            filtered_data = filter_by_mean_deviation(resulting_dataframe, args.deviation_value)
            print(filtered_data)
        elif args.option2:
            filtered_data = filter_by_date_range(resulting_dataframe, args.start_date, args.end_date)
            print(filtered_data)
        elif args.option3:
            filtered_data = group_by_month(resulting_dataframe)
            print(filtered_data)
        elif args.option4:
            create_graph(file_path)
        elif args.option5:
            filtered_data = group_by_month(resulting_dataframe)
            create_monthly_graph(filtered_data, args.month_to_plot)
        else:
            print("Выбран неправильный номер")
    except Exception as ex:
        logging.error(f"option can't be executed: {ex}\n{ex.args}\n")
