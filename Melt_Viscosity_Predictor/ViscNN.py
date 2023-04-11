import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Dropout, Activation, GaussianNoise
from  keras import activations
from keras.regularizers import l2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
import kerastuner as kt
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from utils.metrics import OME


class ViscNN_concat_HP(kt.HyperModel):
    def __init__(self, n_features):
        self.n_features = n_features

    def build(self, hp):
        fp_in = tf.keras.Input(shape=(self.n_features),)
        logMw = tf.keras.Input(shape=(1,))
        log_shear = tf.keras.Input(shape=(1,))
        T = tf.keras.Input(shape=(1,))
        P = tf.keras.Input(shape=(1,))
        fp_size = hp.Int('fp_dense', 30, 150, step = 30)
        fp_dense = Dense(fp_size, activation = 'softplus', kernel_initializer='he_normal')(fp_in)
        merged = Concatenate(axis=1)([fp_dense, logMw, log_shear, T, P])
        L1_size = hp.Int('layer_1', 30, 150, step = 30)
        L2_size = hp.Int('layer_2', 30, 150, step = 30)
        L1_reg = hp.Float('l1_reg', 0.000, 0.0001, step = 0.00001)
        L2_reg = hp.Float('l2_reg', 0.000, 0.0001, step = 0.00001)
        #L3_size = hp.Int('layer_3', 30, 210, step = 30)
        layer_1 = Dense(L1_size, activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(L1_reg))(merged)
        layer_1 = Dropout(hp.Float('dropout_1', 0.1, 0.5, step = 0.05), input_shape = (L1_size,))(layer_1)
        layer_2 = Dense(L2_size, activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(L2_reg))(layer_1)
        layer_2 = Dropout(hp.Float('dropout_2', 0.1, 0.5, step = 0.05), input_shape = (L2_size,))(layer_2)
        #layer_3 = Dense(L3_size, activation='softplus', kernel_initializer='he_normal')(layer_2)
        #layer_3 = Dropout(hp.Float('dropout_3', 0, 0.4, step = 0.1), input_shape = (L3_size,))(layer_3)
        log_eta = Dense(1, activation = activations.tanh)(layer_2)
        model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T, P], outputs=[log_eta])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
        return model

def create_ViscNN_concat(n_features, hp):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw = tf.keras.Input(shape=(1,))
    #logMw_noise = GaussianNoise(0.02)(logMw)
    log_shear = tf.keras.Input(shape=(1,))
    #shear_noise = GaussianNoise(0.02)(log_shear)
    T = tf.keras.Input(shape=(1,))
    #T_noise = GaussianNoise(0.02)(T)
    P = tf.keras.Input(shape=(1,))
    fp_dense = Dense(hp['fp_dense'], activation = 'softplus', kernel_initializer='he_normal')(fp_in)
    merged = Concatenate(axis=1)([fp_dense, logMw, log_shear, T, P])
    layer_1 = Dense(hp['layer_1'], activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(hp['l1_reg']))(merged)
    layer_1 = Dropout(hp['dropout_1'], input_shape = (hp['layer_1'],))(layer_1, training = True)
    layer_2 = Dense(hp['layer_2'], activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(hp['l2_reg']))(layer_1)
    layer_2 = Dropout(hp['dropout_2'], input_shape = (hp['layer_2'],))(layer_2, training = True)
    log_eta = Dense(1, activation = activations.tanh)(layer_2)
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T, P], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model
  



            
class metalearner(tf.keras.Model):
    def __init__(self, CV_models):
        self.models

    def build():
        #for 
        
        predictions = tf.keras.Input(shape=(10,))
        HL = Dense(10, activation='softplus', kernel_initializer='he_normal')(predictions)
        model=tf.keras.models.Model(inputs=[predictions], outputs=[log_eta])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
        return model