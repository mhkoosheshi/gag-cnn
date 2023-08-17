import tensorflow as tf
from tensorflow.keras.losses import Loss


def jaccard_loss(y_pred, y_true, axis=(0, 1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    l = tf.reduce_sum(y_pred * y_pred, axis=axis)
    r = tf.reduce_sum(y_true * y_true, axis=axis)
    jaccard = 1 - (inse + smooth) / (l + r - inse + smooth)
    jaccard = tf.reduce_mean(jaccard)
    return jaccard

def get_loss(loss_name='jaccard_loss'):

    if loss_name=='jaccard_loss':
        return jaccard_loss

    elif loss_name=='mse_loss':
        return tf.keras.losses.MeanSquaredError()
    
    elif loss_name=='mae_loss':
        return tf.keras.losses.MeanAbsoluteError()