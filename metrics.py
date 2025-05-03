import numpy as np
# def dice_coef(y_true, y_pred):
#     return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

# mIoU = MeanIoU(num_classes=2)

# def dice_coef(y_true, y_pred, smooth=1):
#     y_pred_f = K.flatten(y_pred)
#     y_true_f = K.flatten(y_true)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
#     return (2. * intersection + smooth) / union

# def IoU(y_true, y_pred):
#     y_pred_f = K.flatten(y_pred)
#     y_true_f = K.flatten(y_true)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
#     return intersection/union

def class_accuracy(confusion: np.ndarray) -> np.ndarray:
    """
    Return the per class accuracy from confusion matrix.
    Args:
        confusion: the confusion matrix between ground truth and predictions
    Returns:
        a vector representing the per class accuracy
    """
    # extract the number of correct guesses from the diagonal
    preds_correct = np.sum(confusion * np.eye(len(confusion)), axis=-1)
    # extract the number of total values per class from ground truth
    trues = np.sum(confusion, axis=-1)
    # get per class accuracy by dividing correct by total
    return preds_correct / trues

def iou(confusion: np.ndarray) -> np.ndarray:
    """
    Return the per class Intersection over Union (I/U) from confusion matrix.
    Args:
        confusion: the confusion matrix between ground truth and predictions
    Returns:
        a vector representing the per class I/U
    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    # get |intersection| (AND) from the diagonal of the confusion matrix
    intersection = (confusion * np.eye(len(confusion))).sum(axis=-1)
    # calculate the total ground truths and predictions per class
    preds = confusion.sum(axis=0)
    trues = confusion.sum(axis=-1)
    # get |union| (OR) from the predictions, ground truths, and intersection
    union = trues + preds - intersection
    # return the intersection over the union
    return intersection / union

# ALPHA = 0.7
# BETA = 0.3
# GAMMA = 1.33

# def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-5):
#     # flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)

#     inputs = tf.cast(inputs, tf.float32)
#     targets = tf.cast(targets, tf.float32)
    
#     # True Positives, False Positives & False Negatives
#     TP = K.sum((inputs * targets))
#     FP = K.sum(((1-targets) * inputs))
#     FN = K.sum((targets * (1-inputs)))
            
#     Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
#     FocalTversky = K.pow((1 - Tversky), gamma)
    
#     return FocalTversky