import numpy as np
import tensorflow as tf


def OME(y_pred, yy):
    y_pred = tf.cast(tf.math.pow(10.0, y_pred), tf.float32)
    yy = tf.cast(tf.math.pow(10.0, yy), tf.float32)
    div = tf.math.divide_no_nan(y_pred,yy)
    l = tf.math.abs(tf.experimental.numpy.log10(div))
    return tf.math.reduce_mean(l)
#TODO: (1/N).Summation[abs{log(x_pred/x_act)}]

def RMSE(y_pred, yy):
    return np.sqrt(np.sum(np.power((yy-y_pred),2)/len(yy)))

def MSE(y_pred, yy):
    return np.sum(np.power((yy-y_pred),2)/len(yy))

def MAPE(y_pred, yy):
    y_pred = np.power(10, y_pred)
    yy = np.power(10, yy)
    return (1/y_pred.shape[0])*sum([abs((yy[i] - y_pred[i])/yy[i]) for i in range(y_pred.shape[0])])[0]
