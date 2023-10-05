import os
import json
import logging
import requests
from bs4 import BeautifulSoup

with open(os.path.join("Lab1", "fcc.json").replace("\\", "/"), "r") as fcc_file:
    fcc = json.load(fcc_file)

logging.basicConfig(level=logging.INFO)


def create_directory(folder: str) -> str:
    """The function takes the path to the folder and its name,
    checks its presence.
    In case of absence creates a folder.
    """
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except:
        logging.error("Folder don't create", exc_info=True)


def make_lists(url: str) -> list:
    """The function accepts a link to a search query.
    Creates a list that contains links to each object
    from all specified request pages.
    """
    list_url = []
    url_new = url[:-1]
    try:
        for pages in range(fcc.get("PAGES")):
            url_pages: str = f"{url_new}{pages}"
            html = requests.get(url_pages, fcc.get("HEADERS"))
            soup = BeautifulSoup(html.text, "lxml")
            flowers = soup.findAll("img")
            list_url += flowers
        return list_url
    except:
        logging.error("List don't create", exc_info=True)


def download(url_list: list, folder: str) -> str:
    """The function accepts a list of links and a folder name.
    Downloads all images from the list to a folder
    and assigns them a unique number.
    """
    count = 0
    except_count = 0
    for link in url_list:
        if count > fcc.get("MAX_FILES"):
            break
        try:
            src = link["src"]
            print(src)
            response = requests.get(src)
            create_directory(
                os.path.join(fcc.get("MAIN_FOLDER"), folder).replace("\\", "/"))
            try:
                with open(
                        os.path.join(fcc.get("MAIN_FOLDER"), folder, f"{count:04}.jpg").replace("\\", "/"), "wb") as file:
                    file.write(response.content)
                    count += 1
            except:
                logging.error("Uncorrect path")
        except:
            except_count += 1
    logging.info(f"Quantity download files={count}")
    logging.info(f"Quantity uncorrect URL={except_count}")


if __name__ == "__main__":

    r = make_lists(fcc.get("url_rose"))
    download(r, fcc.get("FOLDER_ROSE"))

    t = make_lists(fcc.get("url_tulip"))
    download(t, fcc.get("FOLDER_TULIP"))
