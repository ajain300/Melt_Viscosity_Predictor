import gpflow.mean_functions as mf
import gpflow
import tensorflow as tf
import numpy as np

class mol_weight_mean(mf.MeanFunction):
    def __init__(self, input_dim=None):
        if input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`mol_weight` mean function in combination with expectations."
            )
        self.input_dim = input_dim - 1
        A = np.concatenate((np.array([3.4]), np.zeros(self.input_dim))).reshape(-1,1)
        b = np.zeros((1))
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.b = gpflow.Parameter(b)

    def __call__(self, X):
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b

class shear_mean(mf.MeanFunction):
    def __init__(self, input_dim=None):
        if input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`mol_weight` mean function in combination with expectations."
            )
        self.input_dim = input_dim - 3
        A = np.concatenate((np.array([0,0,-0.6]), np.zeros(self.input_dim))).reshape(-1,1)
        b = np.zeros((1))
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.b = gpflow.Parameter(b)

    def __call__(self, X):
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b

class temp_mean(mf.MeanFunction):
    def __init__(self, temp_pos = 1):
        self.temp = temp_pos
        B = 1
        self.B = gpflow.Parameter(B)

    def __call__(self, X):
        temps = tf.math.add(tf.gather(X, self.temp, axis=1), 275.15)
        return tf.experimental.numpy.log10(tf.reshape(tf.math.exp(tf.math.scalar_mul(self.B, tf.math.reciprocal(temps))), [-1, 1])) #tf.reshape(tf.math.exp(tf.math.reciprocal(temps)), [-1, 1])

class weight_temp_mean(mf.MeanFunction):
    def __init__(self, input_dim = None, temp_pos = 1):
        if input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`mol_weight` mean function in combination with expectations."
            )
        self.input_dim = input_dim - 1
        A = np.concatenate((np.array([3.4]), np.zeros(self.input_dim))).reshape(-1,1)
        b = np.zeros((1))
        self.A = gpflow.Parameter(np.atleast_2d(A))
        self.b = gpflow.Parameter(b)
        self.wM = gpflow.Parameter(0.9)
        self.wT = gpflow.Parameter(0.1)
        self.temp = temp_pos
        B = 1
        self.B = gpflow.Parameter(B)

    def __call__(self, X):
        temps = tf.math.add(tf.gather(X, self.temp, axis=1), 275.15)
        return tf.math.add(tf.math.scalar_mul(self.wM, tf.tensordot(X, self.A, [[-1], [0]]) + self.b), tf.math.scalar_mul(self.wT, tf.experimental.numpy.log10(tf.reshape(tf.math.exp(tf.math.scalar_mul(self.B, tf.math.reciprocal(temps))), [-1, 1]))))
