import os
import logging
import argparse
import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SEARCH_URL = "https://www.bing.com/images/search"

def create_directory(directory: str) -> None:
    """The function creates a folder if it does not exist"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        logger.exception(f"Can't create folder: {exc}")


def download_images(query: str,
                    num_images: int,
                    dataset_directory) -> None:
    """This function uploads images to the selected directory"""   
    create_directory(dataset_directory)
    count = 0
    page = 1
    while count < num_images:
        search_params = {"q": query, "form": "HDRSC2", "first": page}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(SEARCH_URL, params=search_params, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")
        img_tags = soup.find_all('img', {"src": True}, class_='mimg')

        for img_tag in img_tags:
            img_url = img_tag["src"]
            if img_url and img_url.startswith("http"):
                try:
                    img_data = requests.get(img_url, headers=headers, timeout=10).content
                    class_folder = os.path.join(dataset_directory, query)
                    create_directory(class_folder)
                    with open(os.path.join(class_folder, f"{count:04}.jpg"), "wb") as img_file:
                        img_file.write(img_data)
                    count += 1
                    logger.info(f"Downloaded {count}/{num_images} images for '{query}'")
                    if count >= num_images:
                        break
                except Exception as e:
                    logger.exception(f"Error downloading image: {str(e)}")
        page += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images in classes")
    parser.add_argument("classes", nargs='+', default=["cat", "dog"], help="List of classes to download images for.")
    parser.add_argument("--num_images_per_class", type=int, default=1000, help="Number of images to download per class.")
    parser.add_argument("--dataset_directory", default="dataset", help="Path to the dataset directory.")
    args = parser.parse_args()

    for class_name in args.classes:
        download_images(class_name, args.num_images_per_class, args.dataset_directory)
