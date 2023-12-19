import argparse
import logging
import os
import requests
from bs4 import BeautifulSoup
from fake_headers import Headers

BASE_URL = "https://www.bing.com/images/search?q="

logging.basicConfig(level=logging.INFO,
                    filename="logs.log",
                    format="%(levelname)s - %(funcName)s: %(message)s")


def create_dir(folder_path: str, subfolder_path: str) -> None:
    """
    the function creates a main and an additional folder
    Parameters
    ----------
    folder_path : str
    subfolder_path : str
    """
    try:
        subfolder_path = os.path.join(folder_path, subfolder_path)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    except Exception as e:
        logging.exception(f"Can't create a folder: {e}")


def img_download(subfolder_path: str, folder_path: str, num_images: int) -> None:
    """
    the function calls the function to create folders and loads images into them using: 
    "https://www.bing.com/images/"
    Parameters
    ----------
    subfolder_path : str
    folder_path : str
    num_images : int
    """
    create_dir(folder_path, subfolder_path)
    page = 1
    k = 0
    while k < num_images:
        url = f"{BASE_URL}{subfolder_path}&first={page}"
        try:
            headers = Headers(os="mac", headers=True).generate()
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'lxml')
            img_tags = soup.find_all('img', {"src": True}, class_='mimg')
            image_urls = [img['src'] for img in img_tags]

            for img_url in image_urls:
                try:
                    response = requests.get(img_url)
                    filename = os.path.join(folder_path, subfolder_path, f"{k:04}.jpg")

                    with open(filename, 'wb') as f:
                        f.write(response.content)

                    logging.info(f"Downloaded image {k + 1}/{num_images}")
                    k += 1
                except Exception as e:
                    logging.exception(f"Error downloading image: {e}")
            page += 1
        except Exception as e:
            logging.exception(f"Error fetching data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download images from Bing Images')
    parser.add_argument('subfolder_path1', type=str, default='zebra', help='Name of the image 1st class')
    parser.add_argument('subfolder_path2', type=str, default='bay horse', help='Name of the image 2nd class')
    parser.add_argument('folder_path', type=str, default='dataset', help='Output directory for saving images')
    parser.add_argument('num_images', type=int, default=1000, help='Number of images to download')

    args = parser.parse_args()
    img_download(args.subfolder_path1, args.folder_path, args.num_images)
    img_download(args.subfolder_path2, args.folder_path, args.num_images)