from ImageScrapper import Image_Scrapper as IS

MAIN_DIR = "dataset"
POLAR_BEAR_DIR = "polar_beer"
BROWN_BEAR_DIR = "brown_beer"
HEADER = {"User-Agent":"Mozilla/5.0"}
url_polar_bear = "https://www.bing.com/images/search?q=polar_bear.jpg&first=1"
url_brown_bear = "https://www.bing.com/images/search?q=brown_bear.jpg&&first=1"


if __name__ == "__main__":
    pb_scrap = IS(POLAR_BEAR_DIR, MAIN_DIR, url_polar_bear, HEADER)
    bb_scrap = IS(BROWN_BEAR_DIR, MAIN_DIR, url_brown_bear, HEADER) 

    pb_scrap.get_pages_imgs(1050, 50)
    pb_scrap.download() 

    bb_scrap.get_pages_imgs(1050, 50)
    bb_scrap.download()   
     
 





