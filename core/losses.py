#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen


import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


def get_loss(loss_fn, tversky_alpha_beta=None):
    """Wrapper function to allow only storing loss function name in config"""
    if loss_fn == "tversky":
        tversky_function = tversky
        # Update default arguments of tversky() with configured alpha beta.
        if tversky_alpha_beta:
            tversky_function.__defaults__ = tversky_alpha_beta
        return tversky_function
    elif loss_fn == "dice":
        return dice_loss
    else:
        # Used when passing string names of built-in tensorflow optimizers
        return loss_fn


def tversky(y_true, y_pred, alpha=0.40, beta=0.60):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    
    ones = 1 
    p0 = y_pred  # prob that pixels are class i
    p1 = ones - y_pred  # prob that pixels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(p0 * g0)
    fp = alpha * tf.reduce_sum(p0 * g1)
    fn = beta * tf.reduce_sum(p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

def accuracy(y_true, y_pred):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)

def dice_loss(y_true, y_pred):
    """compute dice loss"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_t, y_pred)

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))

# Evaluation metrics

def f_beta(y_true, y_pred, beta=1):
    """compute f-beta score"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    precision = K.sum(tp) / (K.sum(tp) + K.sum(fp))
    recall = K.sum(tp) / (K.sum(tp) + K.sum(fn))
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + K.epsilon())

def f1_score(y_true, y_pred):
    """compute f1 score"""
    return f_beta(y_true, y_pred, beta=1)

def IoU(y_true, y_pred):
    """compute IoU"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fp) + K.sum(fn))

def nominal_surface_distance(y_true, y_pred):
    """compute nominal surface distance"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(fp) / (K.sum(tp) + K.sum(fp) + K.sum(fn))

def Hausdorff_distance(y_true, y_pred):
    """compute Hausdorff distance"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(fp) / (K.sum(tp) + K.sum(fp) + K.sum(fn))

def boundary_intersection_over_union(y_true, y_pred):
    """compute boundary intersection over union"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fp) + K.sum(fn))