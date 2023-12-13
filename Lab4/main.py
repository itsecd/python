import argparse
from func_tasks import(process_dataframe,
                       compute_image_stats,
                       filter_dataframe_by_label,
                       filter_dataframe_by_params,
                       calculate_pixels_stats,
                       generate_histogram)
from open_save import read_csv_to_dataframe, save_to_csv
from graphic import plot_histograms


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--func1", action="store_true", help="Проверка датасета на сбалансированность")
    group.add_argument("--func2", action="store_true", help="Фильтрация датафрейма")
    group.add_argument("--func3", action="store_true", help="Фильтрация датафрейма по макс. ширине, высоте и метке")
    group.add_argument("--func4", action="store_true", help="Группировка датафрейма")
    group.add_argument("--func5", action="store_true", help="Создание гистограммы по рандомному изображению")
    
    parser.add_argument("--input_csv",
                        type=str,
                        default="C:/Users/zhura/Desktop/paths.csv",
                        help="Ввод пути для исходного csv")
    parser.add_argument("--output_csv",
                        type=str,
                        default="C:/Users/zhura/Desktop/processed_data.csv",
                        help="Ввод пути для измененного csv")
    parser.add_argument("--label",
                        type=int,
                        default=0,
                        help="Значение метки для фильтрации")
    parser.add_argument("--max_width",
                        type=int,
                        default=1000,
                        help="Значение максимальной ширины")
    parser.add_argument("--max_height",
                        type=int,
                        default=800,
                        help="Значение максимальной высоты")
    args = parser.parse_args()

    data_frame = read_csv_to_dataframe(args.input_csv)
    processed_df = process_dataframe(data_frame)

    save_to_csv(processed_df, args.output_csv)
    print(f"Данные сохранены в файл {args.output_csv}")

    if args.func1:
        image_stats, label_stats = compute_image_stats(processed_df)

        print("Статистика по размерам изображений:")
        print(image_stats)
        print("\nСтатистика меток класса:")
        print(label_stats)

    if args.func2:
        filtered_data = filter_dataframe_by_label(processed_df, args.label)

        print("\nОтфильтрованный DataFrame по метке:")
        print(filtered_data)

    if args.func3:
        filtered_data = filter_dataframe_by_params(processed_df, args.label, args.max_width, args.max_height)
        
        print("\nОтфильтрованный DataFrame по параметрам:")
        print(filtered_data)

    if args.func4:
        pixels_stats = calculate_pixels_stats(processed_df)

        print("\nСтатистика по количеству пикселей:")
        print(pixels_stats)
        
    if args.func5:
        hist_blue, hist_green, hist_red = generate_histogram(processed_df, args.label)
    
        plot_histograms(hist_blue, hist_green, hist_red)


if __name__ == "__main__":
    main()