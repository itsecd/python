import argparse
from image_scrapper import ImageScrapper

HEADER = {"User-Agent":"Mozilla/5.0"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('request', help='image search query', type=str)
    parser.add_argument('-md', '--maindir', help='specifying the main directory', type=str, default='dataset')
    parser.add_argument('-sd', '--subdir' , help='specifying the subdirectory', type=str, default='data')
    parser.add_argument('-n' , '--number', help='number of downloaded images', type=int, default=1000)
    parser.add_argument('-w' , '--width', help='images width', type=int, default=50)
    args = parser.parse_args()   

    img_scrap = ImageScrapper(args.subdir, args.maindir, args.request, HEADER)
    img_scrap.get_pages_imgs(args.number, args.width)
    img_scrap.download() 
  