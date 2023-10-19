import os
import requests
from bs4 import BeautifulSoup



class Image_Scrapper():
    def __init__(self, _dir : str, _main_dir : str, _url : str, _header:dict[str,str]):
        self.dir = _dir
        self.url = _url
        self.header = _header
        self.main_dir = _main_dir
        self.dataset = set()
    def create_dir(self):
        try:
            os.makedirs(f"{self.main_dir}\{self.dir}")
        except:
            print("[Error]: A folder with the same name already exists")


   
    def get_page_imgs(self, number_page: int, min_width : int) -> list:
        url_new = self.url[:-1] 
        url_pages:str = f"{url_new}{number_page}"
        html = requests.get(url_pages, self.header)
        soup = BeautifulSoup(html.text, "lxml")
        elements = soup.findAll("img")
        imgs = []
        for img in elements:
            if "width" in img.attrs:
                if int(img.attrs["width"]) >= min_width:
                    imgs.append(img)                                        
        return imgs
    
    def get_pages_imgs(self, max_files: int, min_width : int) -> set:
        page = 1
        except_count=0        
        self.create_dir()
        list_response = []
        while len(self.dataset) < max_files:
            src_list = self.get_page_imgs(page, min_width)
            for src in src_list: 
                try:
                    response = requests.get(src["src"])
                    list_response.append(response)                       
                except:
                    except_count+=1
            page += 1
            print(len(self.dataset))
            self.dataset = set(list_response)
        print(f"Quantity ncorrect URL={except_count}")
        return self.dataset  

    def download(self):
        for count, img in enumerate(self.dataset):
            with open(os.path.join( f"{self.main_dir}\{self.dir}\{count:04}.jpg").replace("\\","/"), "wb") as file:
                file.write(img.content)                
    