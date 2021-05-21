import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
from PIL import Image
import requests
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
from io import BytesIO
import argparse
from functools import partial
from multiprocessing import Pool
import logging
import time
import gc
import ast


arg_parser = argparse.ArgumentParser(allow_abbrev=True, description='Download images from url in a directory',)

arg_parser.add_argument('-d','--DIR',required=True,
                       help='Directory name where images will be saved')

arg_parser.add_argument('-c','--CSV',required=True,
                       help='CSV file name which contains the URLs')

arg_parser.add_argument('--url_col',type=str, default='urls',
                       help='Column name which contain the urls')

arg_parser.add_argument('--name_col',type=str, default='id',
                       help='Column name which contains the name. Images will bw saved according to respective entries in that column')

arg_parser.add_argument('-s','--start',type=int,
                       help='Index number of DataFrame Index from where to start downloading')

arg_parser.add_argument('-r','--resize', default=0, type=int,
                       help='Whether to resize the image or not')

arg_parser.add_argument('-rs','--resize_shape',type=str,default='(224,224)',
                       help='resize shape of tuple like 224,224')

arg_parser.add_argument('-w','--workers',type=int,default=4,
                       help='Workers to be used in multiprocessing/ threading')

arg_parser.add_argument('-lt','--log_type',type=str,default='display',
                       help='Error logging method. Write in a file or display. file/print')

args = vars(arg_parser.parse_args())

# Get Argument values
resize = args['resize']
resize_shape = ast.literal_eval(args['resize_shape'])
csv = args['CSV']
DIR = args['DIR']
url_col = args['url_col']
name_col = args['name_col']
workers = args['workers']
log_type = args['log_type']

if log_type == 'file':
    print('\nNOTE:Error logs are stored in the app.log file\n')
    logging.basicConfig(filename='app.log', filemode='a',format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p') # logger config


def load_save_image_from_url(url_img_name_tuple,OUT_DIR,resize=False,resize_shape=(224,224)):
    '''
    Save Images to disc present at a URL. Images are saved in RGB  format.
    args:
        url_img_name_tuple: {tuple (str,str/int)} (URL string where image is present,name of image)
        OUT_DIR: {str} Path to the output directory where you want to store the image
        img_name: {str} Name of image
        resize: {bool} Whether to resize image or not. {default: True}
        resize_shape: {tuple} shape of the resize image {default: (224,224)}
    '''
    url,i = url_img_name_tuple
    try:
        response = requests.get(url)
        try:
            img = Image.open(BytesIO(response.content)) # open image
            img = img.convert('RGB')
            if resize:
                img = img.resize(resize_shape)
            img_format = url.split('.')[-1]
            img_name = str(i)+'.'+img_format
            img.save(OUT_DIR+img_name)

        except Exception as e:
            logging.error(f'Error related to image reading/writing for index {i}: {e}')
            pass

    except (KeyboardInterrupt, SystemExit):
        sys.exit("Forced exit prompted by User. Terminating and joining all threads (if allpicable)")

    except ConnectionError:
        logging.error(f"Connection Error for index {i} ")
        
    except ReadTimeout:
        logging.error(f"Read Timeout for index {i}")
        pass

    except TooManyRedirects:
        logging.error(f"Too many redirects for index {i}")
        pass

    except InvalidURL:
        logging.error(f"Invalid URL at index {i}")
        pass

    except Exception as e:
        logging.error(f'Some Other Error related to REQUESTS: {e}')
        pass


def multiprocessing_download(df,workers,DIR,resize,resize_shape): # if the user prompted multi processing, use this block
    print(f'Multiprocessing with {workers} workers')

    names = df.loc[:,name_col].tolist() # get sliced index
    urls = df.loc[:,url_col].tolist() # get urls
    lis_tups = list(zip(urls,names)) # make a list of 'zipped' list of tuples for Pool

    del df, urls, names
    gc.collect() # free up memory

    pool = Pool(workers) # pool.map only takes 1 iterble
    pool.map(partial(load_save_image_from_url,OUT_DIR=DIR,resize=resize,resize_shape=resize_shape),lis_tups)

    pool.close()
    pool.join()


if __name__ == '__main__':
    start_time = time.time()
    df = pd.read_csv(csv) # read csv
    existing = [f.split('.')[0] for f in listdir(DIR) if isfile(join(DIR, f))] # get existing images
    df = df[~df[name_col].isin(existing)]   

    print(f"""{len(existing)} images present {DIR}. Downloading remaining {df.shape[0]} images from {csv}""")
    
    multiprocessing_download(df,workers,DIR,resize,resize_shape)

    sys.exit(f'Download Completed in *{round(time.time()-start_time,3)}* secs. Exiting script....') # end of program
