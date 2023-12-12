import os
import requests
from bs4 import BeautifulSoup


class ImageScrapper():
    '''This class that searches for a given number of images, a given size, at the specified URL'''

    def __init__(self, dir : str, main_dir : str, querry : str, header : dict[str,str]):
        self._dir = dir
        self._url = f"https://www.bing.com/images/search?q={querry}&&first=1"
        self._header = header
        self._main_dir = main_dir
        self._dirdataset = set()

    def create_dir(self):
        '''This Method that creates a directory for the dataset at the specified address'''
        try:
            os.makedirs(os.path.join(self._main_dir, self._dir))
        except:
            print("[Error]: A folder with the same name already exists")
   
    def get_page_imgs(self, number_page: int, min_width : int) -> list:
        '''This method parses the page and searches for all tags with the value "img" on the page'''
        url_new = self._url[:-1] 
        url_pages:str = f"{url_new}{number_page}"
        html = requests.get(url_pages, self._header)
        soup = BeautifulSoup(html.text, "lxml")
        elements = soup.findAll("img")
        imgs = []
        for img in elements:
            if "width" in img.attrs:
                if int(img.attrs["width"]) >= min_width:
                    imgs.append(img)                                            
        return imgs
    
    def get_pages_imgs(self, max_files: int, min_width : int) -> set:
        '''This method parses the specified number of pages into "img" tags and returns valid image addresses as a set'''
        page = 1
        except_count=0  
        list_response = []
        while len(self._dirdataset) < max_files:
            src_list = self.get_page_imgs(page, min_width)
            for src in src_list: 
                try:
                    if "src" in src:
                        response = requests.get(src["src"])
                        list_response.append(response)                   
                    else:    
                        response = requests.get(src["data-src"])
                        list_response.append(response)                       
                except:
                    except_count+=1
            page += 1            
            self._dataset = set(list_response)
            print(len(self._dataset))
        print(f"Quantity ncorrect URL={except_count}")
        return self._dataset  

    def download(self):
        '''This method downloads files from the collected dataset to the specified directory''' 
        self.create_dir()
        for count, img in enumerate(self._dataset):
            with open(os.path.join( self._main_dir, self._dir, f"{count + 1:04}.jpg"), "wb") as file:
                file.write(img.content)            
