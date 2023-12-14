import os
import logging
import json
from functions import (process_dataframe, extract_image_stats,
                        filter_by_label, filter_by_parameters,
                        group_by_stats, create_histogram)
from open_and_save import read_csv_to_dataframe, save_dataframe_to_csv
from graph import draw_histograms

logging.basicConfig(level=logging.INFO)


def main():
    with open(os.path.join('Lab4', 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    action = config.get('action', None)

    if action is None:
        logging.error("Action not specified in the configuration file")
        return

    data_frame = read_csv_to_dataframe(config.get('input_csv', ''))
    processed_df = process_dataframe(data_frame)

    save_dataframe_to_csv(processed_df, config.get('output_csv', ''))
    logging.info(f"Data saved to file {config.get('output_csv', '')}")

    if action == "check_balance":
        image_stats, label_stats = extract_image_stats(processed_df)
        logging.info("Image size statistics:", image_stats)
        logging.info("\nLabel class statistics:", label_stats)

    elif action == "filter_df_label":
        filtered_data = filter_by_label(processed_df, config.get('label', 0))
        logging.info("\nFiltered DataFrame by label:", filtered_data)

    elif action == "filter_df_params":
        filtered_data = filter_by_parameters(processed_df,
                                                   config.get('label', 0),
                                                   config.get('max_width', 1000),
                                                   config.get('max_height', 800))
        logging.info("\nFiltered DataFrame by parameters:", filtered_data)

    elif action == "grouping":
        pixels_stats = group_by_stats(processed_df)
        logging.info("\nPixel count statistics:", pixels_stats)

    elif action == "make_hist":
        hist_blue, hist_green, hist_red = create_histogram(processed_df, config.get('label', 0))
        draw_histograms(hist_blue, hist_green, hist_red)

    else:
        logging.error("Invalid action specified in the configuration file")


if __name__ == "__main__":
    main()
