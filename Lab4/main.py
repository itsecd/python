import argparse
import logging
import df_functions as df_func
import histogram


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--option1", action="store_true", help="Статистика датафрейма")
    group.add_argument("--option2", action="store_true", help="Фильтрация датафрейма по количеству слов")
    group.add_argument("--option3", action="store_true", help="Фильтрация датафрейма по классу")
    group.add_argument("--option4", action="store_true", help="Группировка датафрейма по классам")
    group.add_argument("--option5", action="store_true", help="Создание гистограммы по датафрейму и заданному классу")

    parser.add_argument("--rate", type=int, default = 1, help="rate of review(from 1 to 5)")
    parser.add_argument("--csv_path", default = "D:/AppProgPython/appprog/csv/2.csv.csv", help="base path of csv file")
    parser.add_argument("--count", type=int, default=150, help="count of words")

    args = parser.parse_args()
    df = df_func.make_dataframe(args.csv_path)

    if args.option1:
        logging.info(f'Статистика датафрема\n{df_func.create_stats(df)}')
    elif args.option2:
        logging.info(f'Фильтрация датафрейма по кол-ву слов\n{df_func.count_filter(df, args.count)}')
    elif args.option3:
        logging.info(f'Фильтрация датафрейма по классу{df_func.class_filter(df, args.rate)}')
    elif args.option4:
        logging.info(f'Группировка датафрейма по классам{df_func.group_by_class(df)}')
    elif args.option5:
        logging.info("Создание гистограммы по датафрейму и заданному классу")
        histogram.build_histogram(df, args.rate)
    else:
        logging.info("Ни одна опция не выбрана")