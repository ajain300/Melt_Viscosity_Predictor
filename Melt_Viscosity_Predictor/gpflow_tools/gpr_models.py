import gpflow
import tensorflow as tf
import numpy as np
import gpflow_tools.mean_func as mf
import gpflow_tools.kernels as kern
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from utils.metrics import OME

def create_GPR(X, Y):
    #k = gpflow.kernels.SquaredExponential( variance = 3.0, lengthscales=20.0) * gpflow.kernels.White(variance = 0.000001)
    k = Tanimoto() * gpflow.kernels.SquaredExponential(variance = 1.0, lengthscales=30.0)
    m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None, noise_variance=7)
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

class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))