import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFont
import os, re, sys, random, shutil, cv2

def clip_tiles(path, save_path, height=1152, width=1152):  
    filenames = os.listdir(path)  
    for file in tqdm(filenames, desc="[Clipping Tiles…]", ascii=False, ncols=75):
        file = file.split('.tif')[0]
        img = cv2.imread(f'{path}{file}.tif', cv2.IMREAD_UNCHANGED)
        # msk = cv2.imread(f'{path}masks/{file}.png')
        row, col = int((((img.shape[0])/height)*2)-1), int((((img.shape[0])/width)*2)-1)
        # row, col = int((img.shape[0])/height), int((img.shape[0])/width)
        count = 1

        for i in range(row):
            for j in range(col):
                crop_img = img[i*(height//2):i*(height//2)+height, j*(width//2):j*(width//2)+width]
                # crop_img = img[i*(height):i*(height)+height, j*(width):j*(width)+width]
                # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                
                # crop_msk = msk[i*(height//2):i*(height//2)+height, j*(width//2):j*(width//2)+width]
                # crop_msk = cv2.cvtColor(crop_msk, cv2.COLOR_BGR2RGB)
                
                # cv2.imwrite(f'{save_path}{file.split("_RGB_nDSM")[0]}_{count}.png', crop_img.astype('uint8'))
                cv2.imwrite(f'{save_path}{file.split("_RGB")[0]}_{count}.png', crop_img.astype('uint8'))
                # cv2.imwrite(f'./masks/{file}_{count}.png', cv2.cvtColor(crop_msk, cv2.COLOR_BGR2RGB))
                count += 1

if __name__ == "__main__":
    path = '../Datasets/Bhopal/roi_7/RGB/'
    save_path = '../Datasets/Bhopal/Bhopal/images_RGB/'
    clip_tiles(path, save_path, height=1152, width=1152)