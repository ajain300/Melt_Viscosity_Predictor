import matplotlib.pyplot as plt
import numpy as np
from os import sys
sys.path.append('../')
from Melt_Viscosity_Predictor.validation.metrics import OME

def val_epochs(history, name,cut = 10, scaler = None):
        
    plt.figure()
    plt.title(name + ' Epoch Training')
    plt.ylabel('MSE Validation Error')
    plt.xlabel('Epochs')
    for hist in history:
        hist_val = hist.history['val_loss'][cut:]
        if scaler is not None:
            hist_val = scaler.inverse_transform(np.array(hist_val).reshape(-1, 1)) - scaler.data_min_
        plt.plot(range(cut,len(hist.history['val_loss'])), hist_val)
    plt.legend(['Fold' + str(i) for i in range(len(history))], loc = (1.04,0))
    plt.show()


def calc_slopes_Mw(X,Y):
    a_2 = (Y[-1] - Y[-5])/(X[-1] - X[-5])
    a_1 = (Y[5] - Y[0])/(X[5] - X[0])
    return a_1, a_2

def compare_cv(models, NN_cv_error, test_in, test_val, scaler):
    test_error = []
    for m in models:
        y_pred = m.predict(test_in)
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
        test_error.append(OME(test_val, y_pred))
    X_axis = np.arange(10)
    #plt.xlabel([f'CV_{i}' for i in range(10)])
    plt.bar(X_axis - 0.15, test_error, 0.3, label = 'Test Error')
    plt.bar(X_axis + 0.15, NN_cv_error, 0.3, label = 'CV_Error')
    return models[test_error.index(min(test_error))]