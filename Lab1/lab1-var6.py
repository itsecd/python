import os
import requests
import logging
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Referer": "https://www.bing.com/",
}
logging.basicConfig(filename="py_log.log", filemode="a", level=logging.INFO)


def create_folder(folder_name: str) -> str:
    """The function creates a subfolder in the dataset
    folder for subsequent downloading of images.
    """
    try:
        if not os.path.exists(f"dataset/{folder_name}"):
            os.mkdir(f"dataset/{folder_name}")
            logging.info("folder created")
    except Exception as e:
        logging.error(f"Error creating folder: {e}")


def create_list(url: str, count_for_list: str, pages=600, width=50) -> list:
    """The function scrolls pages and saves all
    found img tags to a list.
    """
    lst = []
    count = 0
    try:
        for page in range(1, pages+1):
            print(page)
            #if count >= int(count_for_list) or page>=pages:
                #break
            url_pages: str = f"{url[:-1]}{page}"
            response = requests.get(url_pages, headers=HEADERS)
            soup = BeautifulSoup(response.text, "lxml")
            images = soup.find_all("img")
            for img in images:
                if img.get("width") and int(img.get("width")) > width:
                    lst.append(img)
                    count += 1
    except Exception as e:
        logging.error(f"list not created: {e}")
    logging.info("img uploaded to list")
    return lst


def download_images(
    url: str,
    folder_name: str,
    count_for_downloads: str,
    count_for_list: str,
    pages: str,
) -> str:
    """The function searches for links to thumbnails in the list
    and downloads them to a folder.
    In lines 72-74 we skip invalid links.
    There are a lot of incorrect images because we are looking only
    by the src attribute.
    """
    list = create_list(url, count_for_list, pages)
    logging.info("ready for download")
    num = 0
    for img_tag in list:
        if num > int(count_for_downloads):
            break
        try:
            src =img_tag["src"] 
            #if str(src).find("rp") != -1:
                #img_tag += 1
                #break
            response = requests.get(src)
            numbers = format(num).zfill(4)
            create_folder(folder_name)
            try:
                with open(os.path.join("dataset", folder_name, f"{numbers}.jpg"), "wb") as f:
                    f.write(response.content)
                    num += 1
            except Exception as e:
                logging.error(f"Error creating file: {e}")
        except Exception as e:
            logging.error(f"incorrect imgs:{img_tag}, {e}")
    logging.info("Images downloaded")


if __name__ == "__main__":
    logging.info("start")
    os.mkdir("dataset")
    download_images(
        os.getenv("SEARCH_TIGER"),
        os.getenv("FOLDER_TIGER"),
        os.getenv("COUNT_FOR_DOWNLOADS"),
        os.getenv("COUNT_FOR_LIST"),
        600,
    )
    download_images(
        os.getenv("SEARCH_LEOPARD"),
        os.getenv("FOLDER_LEOPARD"),
        os.getenv("COUNT_FOR_DOWNLOADS"),
        os.getenv("COUNT_FOR_LIST"),
        600,
    )
