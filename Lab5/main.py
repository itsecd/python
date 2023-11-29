import argparse
import realization
from grafic import draw_histogram
from open import open_new_csv, save_csv, open_csv


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
        fame = realization.generate_frame(open_new_csv(args.csv_path), "cat")
        save_csv(realization.checbalance_testking_balance(fame), args.new_file_path)
    elif args.option2:
        frame = open_csv(args.csv_path)
        print(realization.filter_by_label(frame, args.label))
    elif args.option3:
        frame = open_csv(args.csv_path)
        print(realization.min_max_filter(frame, args.width, args.height, args.label))
    elif args.option4:
        frame = open_csv(args.csv_path)
        print(realization.grouping(frame))
    elif args.option5:
        frame = open_csv(args.csv_path)
        draw_histogram(realization.histogram_build(frame, args.label))
    else:
        print("Ни одна опция не выбрана")


if __name__ == "__main__":
    main()