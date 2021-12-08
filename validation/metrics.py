import numpy as np
import tensorflow as tf


def OME(y_pred, yy):
    div = tf.math.divide_no_nan(abs(y_pred),abs(yy))
    l = tf.experimental.numpy.log10(tf.math.reduce_prod(div))
    return tf.math.abs(l)

def RMSE(y_pred, yy):
    return np.sqrt(np.sum(np.power((yy-y_pred),2)/len(yy)))

def MSE(y_pred, yy):
    return np.sum(np.power((yy-y_pred),2)/len(yy))

def MAPE(y_pred, yy):
    y_pred = np.power(10, y_pred)
    yy = np.power(10, yy)
    return (1/y_pred.shape[0])*sum([abs((yy[i] - y_pred[i])/yy[i]) for i in range(y_pred.shape[0])])[0]
