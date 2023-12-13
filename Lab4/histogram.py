# histogram.py
import cv2
import matplotlib.pyplot as plt

def plot_histogram(df, class_label):
    random_image_path = df[df['название_класса'] == class_label].sample(1)['абсолютный_путь_к_файлу'].values[0]

    image = cv2.imread(random_image_path)
    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    return hist_b, hist_g, hist_r

def plot_histograms(hist_b, hist_g, hist_r):
    plt.figure(figsize=(12, 6))

    plt.subplot(131)
    plt.plot(hist_b, color='blue')
    plt.title('Гистограмма по синему каналу')
    plt.xlabel('Значение пикселей')
    plt.ylabel('Частота')

    plt.subplot(132)
    plt.plot(hist_g, color='green')
    plt.title('Гистограмма по зеленому каналу')
    plt.xlabel('Значение пикселей')
    plt.ylabel('Частота')

    plt.subplot(133)
    plt.plot(hist_r, color='red')
    plt.title('Гистограмма по красному каналу')
    plt.xlabel('Значение пикселей')
    plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()
