import os
import argparse
import glob

from tqdm import tqdm
from PIL import Image
import numpy as np
from multiprocessing import Pool

from src.detector_class import Detector
import arcface_warping

def resize(img, max_size=1024):
    img_maxsize = max(img.size)
    if img_maxsize <= max_size:
        return img

    scale_k = max_size / img_maxsize
    img = img.resize(((int(img.size[0] * scale_k)),
                     (int(img.size[1] * scale_k))))
    return img

def worker(args):
    i, img_list = args

    detector = Detector()

    for image_name in tqdm(img_list, desc='Process {:2}'.format(i), position = i):

        if image_name.split('.')[-1].lower() not in ['jpeg', 'jpg', 'png']:
            continue

        img_format =  '.' + image_name.split('.')[-1]
        try:
            img = Image.open(image_name).convert('RGB')
            img = resize(img, max_size=1024)
        except:
            continue

        bboxes, landmarks = detector.detect_faces(img)
        if len(bboxes) == 0:
            continue

        bbox, landmarks5 = bboxes[0], landmarks[0]
        warp_img = arcface_warping.preprocess(np.array(img), 
                                              landmarks5.reshape((2,5)).T)

        warp_img = Image.fromarray(warp_img)
        warp_img.save(image_name.replace(img_format, '_warped' + img_format))

def main(conf):
    all_files = glob.glob(os.path.join(conf.list_dir, '*/*.*'))
    worker_lists = []
    for i in range(conf.split_size):
        worker_lists.append(all_files[i::conf.split_size])
    
    print(len(worker_lists))
    with Pool(conf.split_size) as p:
        for _ in p.imap(worker, enumerate(worker_lists)):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector and warping by list') 
    parser.add_argument('--list_dir', 
                        type = str, 
                        help = 'Path to directory with images')
    parser.add_argument('--split_size', 
                        type = int, 
                        help = 'Split size for multiprocess') 

    conf = parser.parse_args()
    main(conf)