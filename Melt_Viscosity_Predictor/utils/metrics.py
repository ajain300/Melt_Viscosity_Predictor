import numpy as np
import tensorflow as tf


def OME(yy, y_pred):
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

def get_CV_error(history, scaler = None):
    cv_error = []
    print(type(history[0]))
    if not hasattr(history[0], 'history'):
        print(history)
        return np.mean(history)*(scaler.data_max_ - scaler.data_min_)[0]
    else: 
        for hist in history:
            hist_val = hist.history['val_loss']
            #print(hist_val)
            if scaler is not None:
                hist_val = hist_val * np.power(scaler.data_max_ - scaler.data_min_,2)
            cv_error.append(hist_val[-1])
        return np.mean(cv_error)