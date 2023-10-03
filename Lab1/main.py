import os
import os.path
import requests
from bs4 import BeautifulSoup

HEADERS={"User-Agent": "Mozilla/5.0"}
MAIN_FOLDER="dataset"
FOLDER_ROSE="rose"
FOLDER_TULIP="tulip"
MAX_FILES=1010
PAGES=150
url_rose="https://www.bing.com/images/search?q=rose.jpg&redig=7D4B2E55AA5E4A4CA223FB76FBC6D258&first=1"
url_tulip="https://www.bing.com/images/search?q=tulip.jpg&qs=UT&form=QBIR&sp=1&lq=0&pq=tulip.jpg&sc=2-9&cvid=9577F520591A403C88168B8637C22677&first=1"



def create_directory(folder: str)->str:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except: 
        print("Folder don't create")

def make_lists(url:str)->list:
    list_url=[]
    url_new=url[:-1] 
    try:
        for pages in range(1, PAGES):
            url_pages:str=f"{url_new}{pages}"
            html = requests.get(url_pages, HEADERS)
            soup = BeautifulSoup(html.text, "lxml")
            flowers = soup.findAll("img")
            list_url += flowers
        return list_url  
    except: 
        print("List don't create") 

def download(url_list: list , folder:str)->str:
    count = 0
    except_count=0
    for link in url_list:
        if count > MAX_FILES:
            break
        try:
            src = link["src"]
            print(src)
            response = requests.get(src)
            create_directory(folder)
            with open(os.path.join(MAIN_FOLDER, folder, f"{count:04}.jpg").replace("\\","/"), "wb") as file:
                file.write(response.content)
                count += 1           
        except:
            except_count+=1
    print(f"Quantity download files={count}") 
    print(f"Quantity ncorrect URL={except_count}")        
            
   
if __name__ == "__main__":

    r=make_lists(url_rose)
    download(r, FOLDER_ROSE)

    t=make_lists(url_tulip)
    download(t, FOLDER_TULIP)
  
