import os
import json
import logging
import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)


def create_directory(folder: str) -> str:
    """The function takes the path to the folder and it's name,
    checks it's presence.
    In case of absence creates a folder.
    """
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except Exception as ex:
        logging.error(f"Couldn't create folder: {ex.message}\n{ex.args}\n")


def make_list(url: str) -> list:
    """The function accepts a link to a search query.
    Creates a list that contains links to each object
    from all specified request pages.
    """
    list_url = []
    try:
        for pages in range(fcc["pages"]):
            url_new = url[:-1]
            url_pages: str = f"{url_new}{pages}"
            html = requests.get(url_pages, fcc['headers'])
            soup = BeautifulSoup(html.text, "lxml")
            flowers = soup.findAll("img")
            list_url += flowers
        return list_url
    except Exception as ex:
        logging.error(f"List don't create: {ex.message}\n{ex.args}\n")


def download(
    max_files: int,
    classes: str,
    url: str,
    main_folder: str,
) -> str:
    """The function accepts number of files, classes (for splitting into folders), url and main-folder name.
    Make a list with all images in classes, downloads from list this images
    and assigns them a unique number.
    """
    count = 0
    except_count = 0
    for c in classes:
        url_list = make_list(url.replace("classes", c))
        for link in url_list:
            count_files = len(os.listdir(os.path.join(main_folder, c)))
            if count_files > max_files:
                count = 0
                continue
            try:
                src = link["src"]
                print(src)
                response = requests.get(src)
                create_directory(os.path.join(
                    main_folder, c).replace("\\", "/"))
                try:
                    with open(os.path.join(main_folder, c, f"{count:04}.jpg").replace("\\", "/"), "wb") as file:
                        file.write(response.content)
                        count += 1
                except Exception as ex:
                    logging.error(f"Uncorrect path: {ex.message}\n{ex.args}\n")
            except Exception as ex:
                except_count += 1
                logging.error(
                    f"Quantity uncorrect URL={except_count}:{src}\n")
        logging.info(
            f"Quantity downloaded files in {c} class is {count_files}")


if __name__ == "__main__":
    with open(os.path.join("Lab1", "fcc.json"), "r") as fcc_file:
        fcc = json.load(fcc_file)

    download(fcc["max_files"], fcc["classes"],
             fcc["search_url"], fcc["main_folder"])
