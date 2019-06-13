import os
import sys
import glob
from tqdm import tqdm as tqdm
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
from skimage import transform as trans

def preprocess(img, landmark):
    image_size = [112,112]
    #if isinstance(img, str):
    #    img = read_image(img)
    
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        src[:,0] += 8.0
    
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    
    assert len(image_size)==2
    
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    
    return warped

'''
def worker(args):
    idx, row = args
    new_crop_path = row.crop_path.replace(pp1, to_pp).replace(pp2, to_pp).replace(pp3, to_pp)
    if os.path.isfile(new_crop_path):
        return new_crop_path

    img = Image.open(row.crop_path)
    if pd.isna(row.x) or pd.isna(row.y):
        bbox_x = img.size[0]//3
        bbox_y = img.size[1]//3
    else:
        bbox_x = int(row.x)
        bbox_y = int(row.y)
        
    landmarks5 = np.zeros((5,2))
    for i, (x,y) in enumerate(zip(['x0', 'x1', 'x2', 'x3', 'x4'],
                                ['y0', 'y1', 'y2', 'y3', 'y4'])):
        landmarks5[i,0] = int(bbox_x + row[x])
        landmarks5[i,1] = int(bbox_y + row[y])
    warp_img = preprocess(np.array(img), landmarks5)

    #new_crop_path = row.crop_path.replace(pp1, to_pp).replace(pp2, to_pp)
    try:
        if not os.path.isdir(os.path.dirname(new_crop_path)):
            os.makedirs(os.path.dirname(new_crop_path))
    except FileExistsError:
        pass
    warp_img = Image.fromarray(warp_img)
    warp_img.save(new_crop_path)
    return new_crop_path
'''