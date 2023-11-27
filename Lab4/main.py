import argparse
import tasks_part
from graphic_part import draw_histogram
from open_save_part import open_new_csv, save_csv, open_csv


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--option1", action="store_true", help="Проверка датасета на сбалансированность")
    group.add_argument("--option2", action="store_true", help="Фильтрация датафрейма")
    group.add_argument("--option3", action="store_true", help="Фильтрация датафрейма по макс. ширине, высоте и метке")
    group.add_argument("--option4", action="store_true", help="Группировка датафрейма")
    group.add_argument("--option5", action="store_true", help="Создание гистограммы по рандомному изображению")

    parser.add_argument("--csv_path", help="base csv-file")
    parser.add_argument("--label", type=int, help="label of image (0 or 1)")
    parser.add_argument("--width", type=int, help="width of image")
    parser.add_argument("--height", type=int, help="height of image")
    parser.add_argument("--class", help="class of image (rose or tulip)")
    parser.add_argument("--new_file_path", help="new_file_path")

    args = parser.parse_args()

    if args.option1:
        dfame = tasks_part.image_forms(open_new_csv(args.csv_path), "rose")
        save_csv(tasks_part.checking_balance(dfame), args.new_file_path)
    elif args.option2:
        dframe = open_csv(args.csv_path)
        print(tasks_part.filter_by_label(dframe, args.label))
    elif args.option3:
        dframe = open_csv(args.csv_path)
        print(tasks_part.min_max_filter(dframe, args.width, args.height, args.label))
    elif args.option4:
        dframe = open_csv(args.csv_path)
        print(tasks_part.grouping(dframe))
    elif args.option5:
        dframe = open_csv(args.csv_path)
        draw_histogram(tasks_part.histogram_build(dframe, args.label))
    else:
        print("Ни одна опция не выбрана")


if __name__ == "__main__":
    main()
