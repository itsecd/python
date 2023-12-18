import json
import logging
import os
import df_works
import hystogram


logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    with open(os.path.join('Lab4', 'settings.json'), 'r') as settings_file:
        settings = json.load(settings_file)

    action = settings.get('action', None)

    if action is None:
        logging.error("Action not specified in the configuration file")
    
    df = df_works.df_build(settings.get('input_csv', ''))
    word_count = settings.get('word_count', None)
    rating = settings.get('rating', None)

    if action == 'stats':
        logging.info(f'Dataframe stats\n{df_works.stats_df(df)}')
    elif action == 'wc_filter':
        logging.info(f'Filter by word count\n{df_works.word_count_filter(df, word_count)}')
    elif action == 'r_filter':
        logging.info(f'Фильтрация датафрейма по классу{df_works.rating_filter(df, rating)}')
    elif action == 'r_group':
        logging.info(f'Группировка датафрейма по классам{df_works.rating_group(df)}')
    elif action == 'hyst':
        logging.info("Создание гистограммы по датафрейму и заданному классу")
        hystogram.build_histogram(df, rating)
    else:
        logging.info("Ни одна опция не выбрана")