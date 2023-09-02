import tensorflow as tf
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
import numpy as np


def jaccard_loss(y_pred, y_true, axis=(0, 1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    l = tf.reduce_sum(y_pred * y_pred, axis=axis)
    r = tf.reduce_sum(y_true * y_true, axis=axis)
    jaccard = 1 - (inse + smooth) / (l + r - inse + smooth)
    jaccard = tf.reduce_mean(jaccard)
    return jaccard

def mae_jaccard_loss(y_pred, y_true, axis=(0, 1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(tf.math.abs(y_pred - y_true), axis=axis)
    l = tf.reduce_sum(y_pred * y_pred, axis=axis)
    r = tf.reduce_sum(y_true * y_true, axis=axis)
    jaccard = (inse + smooth) / (l + r - inse + smooth)
    jaccard = tf.reduce_mean(jaccard)
    # return inse.numpy(), l.numpy(), r.numpy()
    return jaccard

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + 1) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + 1)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def rmse_loss(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def custom_loss(y_true, y_pred):
    
    y_true1 = tf.math.ceil(y_true)*0.9
    y_true1 = 1 - y_true1
    loss = K.mean((K.square(y_pred - y_true))/(y_true1), axis=-1)

    return loss

def get_loss(loss_name='jaccard_loss'):

    if loss_name=='jaccard_loss':
        return jaccard_loss

    elif loss_name=='mse_loss':
        return tf.keras.losses.MeanSquaredError()
    
    elif loss_name=='mae_loss':
        return tf.keras.losses.MeanAbsoluteError()

    elif loss_name=='mape_loss':
        return 0.0001 *tf.keras.losses.MeanAbsolutePercentageError()

    elif loss_name=='mspe_loss':
        return tf.keras.losses.MeanSquaredPercentageError()

    elif loss_name=='mselog_loss':
        return tf.keras.losses.MeanSquaredLogarithmicError()

    elif loss_name=='tversky_loss':
        return tversky

    elif loss_name=='focal_tversky_loss':
        return focal_tversky

    elif loss_name=='rmse_loss':
        return rmse_loss

    elif loss_name=='custom_loss':
        return custom_loss