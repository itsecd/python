import os
import requests
from bs4 import Beautifulsoup


MAIN_DIR = "dataset"
POLAR_BEAR_DIR = "polar_beer"
BROWN_BEAR_DIR = "brown_beer"
HEADER = {"User-Agent":"Mozilla/5.0"}
url_polar_bear = "https://yandex.ru/images/search?text=polar%20bear"
url_brown_bear = "https://yandex.ru/images/search?text=brown%20bear"

class Image_Scrapper():
    def __init__(self, _dir : str, _url : str, _header:dict[str,str]):
        self.dir = _dir
        self.url = _url
        self.header = _header
    def create_dir():
        pass
    def create_list():
        pass
    def download():
        pass

if __name__ == "__main__":
    pb_scrap = Image_Scrapper(POLAR_BEAR_DIR, url_polar_bear, HEADER)
    bb_scrap = Image_Scrapper(BROWN_BEAR_DIR, url_brown_bear, HEADER) 
    pb_scrap.download()   
    bb_scrap.download()    





