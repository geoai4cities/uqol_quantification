from libraries import *
from metrics import *
from rgb2onehot import *
from paths import *
from hyperparameters import *
from data_generators import *
from deeplabv3plus import *
from callbacks import *
from plot_training_curves import *
from compute_test_metrics import *
from inception_resnetv2_unet import *
from vgg19_unet import *

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.18)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# img_dir = '../Bhopal/Sample_Patches/patch_1_rg_nDSM_nir/'
# save_dir = '../Bhopal/Sample_Patches/patch_1_rg_nDSM_nir_int/'
# img_dir = save_dir 
# save_pred_dir = '../Output/predict_on_tiles/preds/'
# save_stitch_dir = '../Output/predict_on_tiles/stitch_preds/'

def convert_datatype(img_dir, save_dir, height=256, width=256):
    filenames = os.listdir(img_dir)
    for file in tqdm(filenames, desc="[Converting…]", ascii=False, ncols=75):
        # file = file.split('.tif')[0]
        img = cv2.imread(f'{img_dir}{file}', cv2.IMREAD_UNCHANGED)
        cv2.imwrite(f'{save_dir}{file}', img.astype('uint8'))

def load_model(weights_path):
    model = vgg19_unet(shape = (img_size, img_size, 4), num_classes = 2)
    # model = inception_resnetv2_unet(shape = (img_size, img_size, 4), num_classes = 2)
    # model = DeepLabV3Plus(shape = (img_size, img_size, 4), num_classes = 2)
    model.compile(optimizer=Adam(learning_rate = lr), loss=loss, metrics=[dice_coef, IoU, "accuracy"])
    model.load_weights(f'{weights_path}')

    return model

def predict_on_tiles(tiles_dir, model, save_pred_dir, gray=False):
    files = os.listdir(tiles_dir)
    count = 1
    for tile in tqdm(files, desc="[Predicting…]", ascii=False, ncols=75):
        # img = cv2.imread(f'{tiles_dir}{tile}', cv2.IMREAD_UNCHANGED)
        # # img = (np.array(img)/255).astype('uint8')
        # img = ((np.array(img)).astype('uint8'))/255
        # img = np.expand_dims(img, axis=0)

        img = image.load_img(f'{tiles_dir}{tile}', color_mode='rgba', target_size=(256, 256))
        img = image.img_to_array(img)
        # img = img.astype('uint8') # added later
        img = np.expand_dims(img, axis=0)
        img = img/255
        # print(img.shape)

        pred_img = model.predict(img, verbose=0)
        pred_img = onehot_to_rgb(pred_img[0], id2code)
        pred_filename = tile.split('.tif')[0]
        if gray == True:
            pred_img[pred_img == 255] = 1
            cv2.imwrite(f'{save_pred_dir}{pred_filename}.png', cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY))
        else:
            cv2.imwrite(f'{save_pred_dir}{pred_filename}.png', cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        
        count += 1

def stitch_tiles(pred_tiles_dir, save_dir, file_name):
#     !mkdir hconcat_tiles
    img_list = imread_collection(f'{pred_tiles_dir}*.png')
    j = 0
    for i in range(0, len(os.listdir(pred_tiles_dir)), 100):
        img = cv2.hconcat(img_list[i:i+100])
        cv2.imwrite(f'{save_dir}/hconcat_tile_{j}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        j += 1
                   
    hconcat_imgs = imread_collection(f'{save_dir}/*.png')
    print(len(hconcat_imgs))
    final_pred_tile = cv2.vconcat(hconcat_imgs)
    cv2.imwrite(f'{save_dir}../{file_name}', cv2.cvtColor(final_pred_tile, cv2.COLOR_BGR2GRAY))
    print(final_pred_tile.shape)
    return final_pred_tile
        


# convert_datatype(img_dir=img_dir, save_dir=save_dir, height=256, width=256)


# tiles_dir = '../Bhopal/Sample_Patches/patch_1_rg_nDSM_nir_int/'
# save_pred_dir = '../Results/RG_nDSM_NIR/vgg19_unet/patch_1/'
# save_dir = '../Results/RG_nDSM_NIR/vgg19_unet/concat_predictions/'
# filename = 'patch_1_pred.png'

# vgg19_unet_path = '../Output/vgg19_unet/exp_4/'

# model = load_model(weights_path=f'{vgg19_unet_path}rgbd/model.h5')
# predict_on_tiles(tiles_dir=tiles_dir, model=model, save_pred_dir=save_pred_dir)
# stitch_tiles(pred_tiles_dir=save_pred_dir, save_dir=save_dir, file_name=filename)



# tiles_dir = '../Bhopal/Sample_Patches/patch_1_rg_nDSM_nir_int/'
# save_pred_dir = '../Results/RG_nDSM_NIR/deeplabv3plus/patch_1/'
# save_dir = '../Results/RG_nDSM_NIR/deeplabv3plus/concat_predictions/'
# filename = 'patch_1_pred.png'

# deeplabv3plus_path = '../Output/deeplabv3plus/exp_8/'

# model = load_model(weights_path=f'{deeplabv3plus_path}rgbd/model.94-0.04.h5')
# predict_on_tiles(tiles_dir=tiles_dir, model=model, save_pred_dir=save_pred_dir)
# stitch_tiles(pred_tiles_dir=save_pred_dir, save_dir=save_dir, file_name=filename)




# model = load_model(weights_path=f'{vgg19_unet_path}rgdn/model.h5')
# predict_on_tiles(tiles_dir=roi_1, model=model, save_pred_dir=roi_1_pred)
# stitch_tiles(pred_tiles_dir=roi_1_pred, save_dir=roi_1_concat_pred, file_name='roi_1_pred.png')


# model = load_model(weights_path=f'{deeplabv3plus_path}rgb/model.h5')
# predict_on_tiles(tiles_dir=img_dir, model=model, save_pred_dir=save_pred_dir)
# stitch_tiles(pred_tiles_dir=save_pred_dir, save_dir=save_stitch_dir)
