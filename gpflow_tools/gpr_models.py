import gpflow
import tensorflow as tf
import numpy as np
import gpflow_tools.mean_func as mf
import gpflow_tools.kernels as kern
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from Melt_Viscosity_Predictor.validation.metrics import OME, MSE

def create_GPR(X, Y):
    k = gpflow.kernels.Matern52()
    m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    return m

def train_GPR(XX, Y_tot, M = None, S = None, T = None, LOO = False):
    n_splits = Y_tot.shape[0] if LOO else 10
    kf = KFold(n_splits=n_splits, shuffle = True)
    models = []
    error = []
    X_tot = np.concatenate((XX, M, S, T), axis = 1) if M else XX
    for train_index, test_index in kf.split(XX):
        X, xx = X_tot[train_index], X_tot[test_index]
        Y, yy = Y_tot[train_index], Y_tot[test_index]
        models.append(create_GPR(X, Y))
        m = models[-1]
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        mean, var = m.predict_y(xx)
        error.append(OME(tf.cast(mean, tf.float32), yy))
        mean_train, var_train = m.predict_y(X)
    return models[error.index(min(error))], error
