from metrics import *
from rgb2onehot import *

import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from random import shuffle
from natsort import natsorted
import matplotlib.pyplot as plt
import os, random, cv2, argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score,jaccard_score


def compute_metrics(true_msk_path, pred_msk_path):
    Y_true_all, y_pred_all = np.array([]), np.array([])
    filenames = natsorted(os.listdir(true_msk_path))

    for file in tqdm(filenames, desc="[Computing…]", ascii=False, ncols=75):
        # true_msk, pred_msk = Image.open(f'{true_msk_path}{file}'), Image.open(f'{pred_msk_path}{file}')
        true_msk, pred_msk = cv2.imread(f'{true_msk_path}{file}'), cv2.imread(f'{pred_msk_path}{file}')
        true_msk = cv2.cvtColor(true_msk, cv2.COLOR_BGR2RGB)
        pred_msk = cv2.cvtColor(pred_msk, cv2.COLOR_BGR2RGB)

        true_msk, pred_msk = rgb_to_onehot(true_msk, id2code), rgb_to_onehot(pred_msk, id2code)

        Y_true = np.argmax(true_msk, axis=-1) # Convert one-hot to index
        y_pred = np.argmax(pred_msk, axis=-1) # Convert one-hot to index

        Y_true_flat = Y_true.flatten()
        y_pred_flat = y_pred.flatten()
            
        Y_true_all = np.append(Y_true_all, Y_true_flat)
        y_pred_all = np.append(y_pred_all, y_pred_flat)

    print('\n\n')
    print(classification_report(Y_true_all, y_pred_all))
    print('\n\n')

    cm = confusion_matrix(Y_true_all, y_pred_all)
    df_cm = pd.DataFrame(cm, label_names, label_names)

    print('IoU', np.sum(iou(cm))/6)
    print('IoU', jaccard_score(Y_true_all, y_pred_all, average='macro'))
    print('IoU', jaccard_score(Y_true_all, y_pred_all, average=None))
    print('Accuracy:', accuracy_score(Y_true_all, y_pred_all))
    print('Precision (Weighted):', precision_score(Y_true_all, y_pred_all, average='weighted'))
    print('Precision (Macro):', precision_score(Y_true_all, y_pred_all, average='macro'))
    print('Recall (Weighted):', recall_score(Y_true_all, y_pred_all, average='weighted'))
    print('Recall (Macro):', recall_score(Y_true_all, y_pred_all, average='macro'))
    print('F1 Score (Weighted):', f1_score(Y_true_all, y_pred_all, average='weighted'))
    print('F1 Score (Macro):', f1_score(Y_true_all, y_pred_all, average='macro'))

    # Class wise scores
    per_class_acc = class_accuracy(cm)
    per_class_iou = iou(cm)
    per_class_f1 = f1_score(Y_true_all, y_pred_all, average=None)

    per_class_df = pd.DataFrame(zip(label_names, per_class_acc.T), columns=['Class', 'Accuracy'])
    per_class_df['IoU'] = per_class_iou
    per_class_df['F1'] = per_class_f1

    print('\n\n')
    print(per_class_df)

    # fig, ax = plt.subplots(figsize=(7,6))
    # sns.set(font_scale=1.4) # for label size
    # sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.YlGnBu)
    # plt.title('Confusion Matrix for DeepLabV3+ CNN\n', fontsize=12)
    # plt.savefig(f'./test_results/exp_5/OSA_True_Learnable_Resizer.png', transparent= False, bbox_inches= 'tight', dpi= 300)
    # plt.savefig(f'./test_results/exp_6/RGB.png', transparent= False, bbox_inches= 'tight', dpi= 300)

# compute_metrics(true_msk_path='./datasets/test_data_v1/masks/', pred_msk_path='./test_results/exp_6/RGB/epoch_60/')

 
# compute_metrics(true_msk_path='./datasets/test_data_v2/masks/', pred_msk_path='./test_results/v2/exp_1/RGB/')
# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v4/exp_1/RGB_DSM/')

# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v6/exp_2/OSA_True/')
# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v4/exp_2/OSA_False/')

# compute_metrics(true_msk_path='./datasets/test_data_v2/masks/', pred_msk_path='./test_results/v3/exp_3/OSA_True/')
# compute_metrics(true_msk_path='./datasets/test_data_v2/masks/', pred_msk_path='./test_results/v2/exp_3/OSA_False/')

# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v6/exp_4/OSA_True_Learnable_Resizer/')

# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v4/exp_5/OSA_True_Learnable_Resizer/')

# compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v7/exp_1/RGB_nDSM/')

compute_metrics(true_msk_path='./datasets/test_data_v3/masks/', pred_msk_path='./test_results/v7/exp_1/RGB_nDSM_V2/')

# def compute_metrics(model, test_datagen, steps, test_eval, path):
#     Y_true_all, y_pred_all = np.array([]), np.array([])
#     count = 0

#     for i in range(steps):
#         batch_img, batch_mask = next(test_datagen)
#         pred_all= model.predict(batch_img)
        
#         for j in range(0,np.shape(pred_all)[0]):
#             count += 1
#             true_msk = batch_mask[j]
#             pred_msk = pred_all[j]
#             Y_true = np.argmax(true_msk, axis=-1) # Convert one-hot to index
#             y_pred = np.argmax(pred_msk, axis=-1) # Convert one-hot to index

#             Y_true_flat = Y_true.flatten()
#             y_pred_flat = y_pred.flatten()
                
#             Y_true_all = np.append(Y_true_all, Y_true_flat)
#             y_pred_all = np.append(y_pred_all, y_pred_flat)

#     # Print metrics
#     print('\n\n')
#     print(classification_report(Y_true_all, y_pred_all))
#     print('\n\n')
#     print('IoU', test_eval['IoU'])
#     print('Accuracy:', accuracy_score(Y_true_all, y_pred_all))
#     print('Precision (Weighted):', precision_score(Y_true_all, y_pred_all, average='weighted'))
#     print('Precision (Macro):', precision_score(Y_true_all, y_pred_all, average='macro'))
#     print('Recall (Weighted):', recall_score(Y_true_all, y_pred_all, average='weighted'))
#     print('Recall (Macro):', recall_score(Y_true_all, y_pred_all, average='macro'))
#     print('F1 Score (Weighted):', f1_score(Y_true_all, y_pred_all, average='weighted'))
#     print('F1 Score (Macro):', f1_score(Y_true_all, y_pred_all, average='macro'))

#     # Plot confusion matrix
#     cm = confusion_matrix(Y_true_all, y_pred_all)
#     df_cm = pd.DataFrame(cm, label_names, label_names)
#     fig, ax = plt.subplots(figsize=(7,6))
#     # sns.set(font_scale=1.4) # for label size
#     sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.YlGnBu)
#     plt.title('Confusion Matrix for DeepLabV3+ CNN\n', fontsize=12)
#     plt.savefig(f'{path}confusion_matrix.png', transparent= False, bbox_inches= 'tight', dpi= 300)
    
#     # Class wise scores
#     per_class_acc = class_accuracy(cm)
#     per_class_iou = iou(cm)
#     per_class_f1 = f1_score(Y_true_all, y_pred_all, average=None)

#     per_class_df = pd.DataFrame(zip(label_names, per_class_acc.T), columns=['Class', 'Accuracy'])
#     per_class_df['IoU'] = per_class_iou
#     per_class_df['F1'] = per_class_f1

#     print('\n\n')
#     print(per_class_df)

#     # return Y_true_all, y_pred_all