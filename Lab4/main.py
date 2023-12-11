from func_tasks import(process_dataframe,
                       compute_image_stats,
                       filter_dataframe_by_label,
                       filter_dataframe_by_params,
                       calculate_pixels_stats,
                       generate_histogram)
from open_save import read_csv_to_dataframe, save_to_csv
from graphic import plot_histograms


def main():
    csv_file = input("Введите путь к CSV файлу: ")
    output_csv = input("Введите путь для сохранения обработанных данных: ")

    data_frame = read_csv_to_dataframe(csv_file)
    processed_df = process_dataframe(data_frame)

    save_to_csv(processed_df, output_csv)
    print(f"Данные сохранены в файл {output_csv}")
    image_stats, label_stats = compute_image_stats(processed_df)

    print("Статистика по размерам изображений:")
    print(image_stats)
    print("\nСтатистика меток класса:")
    print(label_stats)
    label=0
    filtered_data = filter_dataframe_by_label(processed_df, label)

    print("\nОтфильтрованный DataFrame по метке:")
    print(filtered_data)

    filtered_data = filter_dataframe_by_params(processed_df, label='polar bear', max_width=1000, max_height=800)
    print("\nОтфильтрованный DataFrame по параметрам:")
    print(filtered_data)
    pixels_stats = calculate_pixels_stats(processed_df)

    print("\nСтатистика по количеству пикселей:")
    print(pixels_stats)
    label = 'polar bear'
    hist_blue, hist_green, hist_red = generate_histogram(processed_df, label)
    
    plot_histograms(hist_blue, hist_green, hist_red)


if __name__ == "__main__":
    main()