import logging
import matplotlib.pyplot as plt
import pandas as pd

from df_functions import group_by_class, make_dataframe



logging.basicConfig(level=logging.INFO)



def build_histogram(df: pd.DataFrame, class_label: int) -> None:
    try: 
        df.loc[:, 'Рейтинг'] = pd.to_numeric(df['Рейтинг'])
        plt.plot(df['Рейтинг'], df[''])

    except Exception as exc:
        logging.error(f"Can not build histogram: {exc}\n{exc.args}\n")




