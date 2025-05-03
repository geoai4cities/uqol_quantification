import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFont
import os, re, sys, random, shutil, cv2

def stack_dsm(img_path, dsm_path, save_path):
    filenames = os.listdir(img_path)
    for file in tqdm(filenames, desc="[Stacking DSM…]", ascii=False, ncols=75):
        img = cv2.imread(f'{img_path}{file}', cv2.COLOR_BGR2RGB)
        dsm = cv2.imread(f'{dsm_path}{file}', cv2.COLOR_BGR2GRAY)
        img_rgbd = cv2.merge([img, dsm])
        cv2.imwrite(f'{save_path}{file}', img_rgbd.astype('uint8'))

if __name__ == "__main__":
    img_path = './datasets/PotsdamRGB/images/'
    dsm_path = './datasets/PotsdamRGB/dsms/'
    save_path = './datasets/PotsdamRGB_DSM/images/'
    stack_dsm(img_path, dsm_path, save_path)
