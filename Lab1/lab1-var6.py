import os
import requests
from bs4 import BeautifulSoup

#function for creating folders

def create_folders(folder_name_first,folder_name_second):
        os.mkdir('dataset')
        try:
          os.mkdir(f'dataset/{folder_name_first}')
          os.mkdir(f'dataset/{folder_name_second}')
        except:
          print("Problem! Try again")

#function for downloading images

def download_images(folder_name,split):
  k=0
  for p in range(0,5):
    urls=f"https://www.yandex.ru/images/search?lr=51&p={p}&rpt=image&{split}"
    r = requests.get(urls,headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(r.text, 'lxml')
    images = soup.find_all('img')
    if(len(images)==2):
        print("Houston, we have a problem")
        continue
    else:
        count=0
        i=k-1
        for image in images[1:]:
          src="https:"+image['src']
          response=requests.get(src, headers={"User-Agent":"Mozilla/5.0"})
          with open(f"dataset/{folder_name}/{format(i).zfill(4)}.jpg",'wb') as f:
              f.write(response.content)
              count+=1
              i+=1
        k+=(i-1) 
        print(f"Total {k} Image Found!")
   
#calling functions
url_tiger='https://yandex.ru/images/search?text=tiger'
url_leopard='https://yandex.ru/images/search?text=leopard'
split_tiger=(url_tiger.split('?'))[1]
split_leopard=(url_leopard.split('?'))[1]
folder_name_tiger="tigers"
folder_name_leopard="leopards"
create_folders(folder_name_tiger,folder_name_leopard)
download_images(folder_name_tiger,split_tiger)
download_images(folder_name_leopard,split_leopard)


