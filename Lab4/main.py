import argparse
import logging
import func
import histogram

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Обработка данных отзывов")
    parser.add_argument("--option1", action="store_true", help="Получить статистику датафрейма")
    parser.add_argument("--option2", action="store_true", help="Фильтрация датафрейма по количеству слов")
    parser.add_argument("--option3", action="store_true", help="Фильтрация датафрейма по классу")
    parser.add_argument("--option4", action="store_true", help="Группировка датафрейма по классам")
    parser.add_argument("--option5", action="store_true", help="Создание гистограммы для заданного класса")

    parser.add_argument("--rate", type=str, default="good", help="Оценка отзыва")
    parser.add_argument("--csv_path", default="P:/python-v8/normal_dataset.csv", help="Путь к файлу CSV")
    parser.add_argument("--count", type=int, default=150, help="Количество слов")

    args = parser.parse_args()
    df = func.load_reviews(args.csv_path)

    if args.option1:
        logging.info(f'Статистика датафрейма:\n{func.get_reviews_statistics(df)}')
    elif args.option2:
        logging.info(f'Фильтрация датафрейма по количеству слов:\n{func.filter_reviews_by_word_count(df, args.count)}')
    elif args.option3:
        logging.info(f'Фильтрация датафрейма по классу:\n{func.filter_reviews_by_class(df, args.rate)}')
    elif args.option4:
        logging.info(f'Группировка датафрейма по классам:\n{func.group_reviews_by_class(df)}')
    elif args.option5:
        logging.info("Создание гистограммы для заданного класса")
        histogram.plot_word_frequency(df, args.rate)
    else:
        logging.info("Не выбрана ни одна опция")

if __name__ == "__main__":
    main()