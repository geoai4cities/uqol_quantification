from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from skimage.io import imread_collection

from glob import glob
from random import shuffle
import argparse
import os, random, cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
import time
import copy
import logging
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.autograd import Variable
from core.configs.default_seg import _C as cfg
from core.utils.utils import weights_init_normal, UnNormalize, adjust_param, setup_logger, _data_part
from core.models.focal_loss import FocalLoss2d
from core.models.build import build_seg_model
from core.models.output_discriminator import OutputDiscriminator
from core.datasets.seg_dataset import SegDataset
from core.utils.utils import *
from evaluate import evl
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

class_dict_df = pd.read_csv(f'./class_dict.csv', index_col=False, skipinitialspace=True)
label_names= list(class_dict_df['class'])
label_codes = []
r= np.asarray(class_dict_df.r)
g= np.asarray(class_dict_df.g)
b= np.asarray(class_dict_df.b)

for i in range(len(class_dict_df)):
    label_codes.append(tuple([r[i], g[i], b[i]]))
    

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}

name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}

def convert_file(img_path, save_img_path):
    filenames = natsorted(os.listdir(img_path))
    for file in tqdm(filenames, ncols=70):
        image_orig = cv2.imread(f'{img_path}{file}', cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f'{save_img_path}{file.split(f".tif")[0]}.png', image_orig)

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def save_predictions(model_path, in_channels, target_dataset_path, save_path):
    model = smp.DeepLabV3(
        encoder_name='resnet34',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,  # model output channels (number of classes in your dataset)
    )
    device = 'cuda'
    # model.to(device)
    model.load_state_dict(torch.load(f"{model_path}"))

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
        # transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))
    ])

    # filenames = natsorted(os.listdir(target_dataset_path))
    # filenames = os.listdir(target_dataset_path)
    
    filenames = glob(f'{target_dataset_path}*')
    # random.seed(42)
    # shuffle(filenames)

    for file in tqdm(filenames, ncols=70):
        image_orig = Image.open(f'{file}')
        # image_orig = cv2.imread(f'{file}', cv2.IMREAD_UNCHANGED)
        image = preprocess(image_orig).unsqueeze(0)

        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            pred_mask = model(image)

        # Thresholding: Assign the class with the highest probability
        _, pred_mask_one_hot = pred_mask[0].max(dim=0)

        # Convert to one-hot encoding
        num_classes = pred_mask[0].shape[0]
        pred_mask_one_hot = torch.nn.functional.one_hot(pred_mask_one_hot, num_classes=num_classes)


        # cv2.imwrite(f'{save_path}{file.split(f"{target_dataset_path}")[1]}', cv2.cvtColor(onehot_to_rgb(pred_mask_one_hot, id2code), cv2.COLOR_RGB2BGR))

        pred_img = np.argmax(pred_mask_one_hot, axis=-1).cpu().detach().numpy()
        # print(pred_img.shape)
        cv2.imwrite(f'{save_path}{file.split(f"{target_dataset_path}")[1]}', pred_img)

def stitch_tiles(pred_tiles_dir, save_dir, file_name):
#     !mkdir hconcat_tiles
    img_list = imread_collection(f'{pred_tiles_dir}*.png')
    # j = 0
    # for i in tqdm(range(0, len(os.listdir(pred_tiles_dir)), 69), ncols=70):
    #     img = cv2.hconcat(img_list[i:i+69])
    #     cv2.imwrite(f'{save_dir}/hconcat_tile_{j}.png', img)
    #     j += 1
                   
    hconcat_imgs = imread_collection(f'{save_dir}/*.png')
    print(len(hconcat_imgs))
    final_pred_tile = cv2.vconcat(hconcat_imgs)
    # cv2.imwrite(f'{save_dir}../{file_name}', cv2.cvtColor(final_pred_tile, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(f'{save_dir}../{file_name}', final_pred_tile)
    print(final_pred_tile.shape)
    # return final_pred_tile

# convert_file(img_path='../Bhopal/tile/', save_img_path='../Bhopal/tile_png/')
# save_predictions(model_path='./res/exp_2/model/model_OSA_True.pt', in_channels=4, target_dataset_path='../Bhopal/tile_png/', save_path='../Bhopal/tile_pred/')

stitch_tiles(pred_tiles_dir='../Bhopal/tile_pred/', save_dir='../Bhopal/tile_pred_hconcat/', file_name='Bhopal_LULC_Map.tif')