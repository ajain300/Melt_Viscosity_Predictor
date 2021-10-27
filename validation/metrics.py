import numpy as np
import tensorflow as tf

def RMSE(y_pred, yy):
    return np.sqrt(np.sum(np.power((yy-y_pred),2)/len(yy)))

def MAPE(y_pred, yy):
    y_pred = np.power(10, y_pred)
    yy = np.power(10, yy)
    return (1/y_pred.shape[0])*sum([abs((yy[i] - y_pred[i])/yy[i]) for i in range(y_pred.shape[0])])[0]
