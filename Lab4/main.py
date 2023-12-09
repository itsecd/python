import cv2
import logging
import pandas as pd
import matplotlib.pyplot as plt 

logging.basicConfig(level=logging.INFO)


def histogram(frame: pd.DataFrame, label: int) -> list:
    '''строит гистограмму с использованием средств библиотеки OpenCV'''
    try:
        image = filter(frame, label)["Absolute path"].sample()
        image_bgr = cv2.imread(image.values[0])
        height, width, depth = image_bgr.shape
        b, g, r = cv2.split(image_bgr)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]) / (height * width)
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]) / (height * width)
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]) / (height * width)
        hists = [hist_b, hist_g, hist_r]
        return hists
    except:
        logging.error(f"Error in histogram\n")


def draw_histogram(hists: list) -> None:
    '''отрисовка гистограмм'''
    colors = ["blue", "green", "red"]
    for i in range(len(hists)):
        plt.plot(hists[i], color=colors[i], label=f"Histogram_{colors[i]}")
        plt.xlim([0, 256])
    plt.ylabel("density")
    plt.title("Histograms BGR")
    plt.xlabel("range")
    plt.legend()
    plt.show()


def open_new_csv(csv_path: str) -> pd.DataFrame:
    '''открытие файла для именования столбцов и удаления Relative path'''
    frame = pd.read_csv(
        csv_path, delimiter=",", names=["Absolute path", "Relative path", "Class"]
    )
    frame_copy = frame.drop("Relative path", axis=1)
    return frame_copy


def save_csv(frame: pd.DataFrame, file_path: str) -> None:
    '''сохранение DataFrame в файл'''
    frame.to_csv(file_path, index=False)
    logging.info(f"Действие успешно выполнено! \n")

