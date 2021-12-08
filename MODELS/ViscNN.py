import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K

def log10(x):
    numerator = K.log(x)
    denominator = K.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def create_ViscNN(n_features = 158):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw = tf.keras.Input(shape=(1,))
    log_shear = tf.keras.Input(shape=(1,))
    T = tf.keras.Input(shape=(1,))
    layer_1 = Dense(128, activation='relu', kernel_initializer='he_normal')(fp_in)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    params_layer = Dense(5, activation= None)(layer_2)
    A,a,B,To,n = tf.keras.layers.Lambda(lambda x: tf.split(x,[1,1,1,1,1],axis=-1))(params_layer)
    log_eta = tf.keras.layers.Lambda(lambda x: x[0] + x[1]*x[5] + x[4]*x[6] + (x[2]/(x[7] - x[3])))([A,a,B,To,n, logMw, log_shear, T])
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.01), loss='mse')
    return model

def create_ViscNN_concat(n_features = 158):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw = tf.keras.Input(shape=(1,))
    log_shear = tf.keras.Input(shape=(1,))
    T = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, logMw, log_shear, T])
    layer_1 = Dense(128, activation='relu', kernel_initializer='he_normal')(merged)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    params_layer = Dense(5, activation='relu')(layer_2)
    log_eta = Dense(1, activation = None)(params_layer)
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.01), loss='mse')
    return model

def create_ViscNN_comb(n_features):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw_norm = tf.keras.Input(shape=(1,))
    log_shear_norm = tf.keras.Input(shape=(1,))
    T_norm = tf.keras.Input(shape=(1,))
    logMw = tf.keras.Input(shape=(1,))
    log_shear = tf.keras.Input(shape=(1,))
    T = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, logMw_norm, log_shear_norm, T_norm])
    layer_1 = Dense(128, activation='relu', kernel_initializer='he_normal')(merged)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    params_layer = Dense(5, activation= None)(layer_2)
    A,a,B,To,n = tf.keras.layers.Lambda(lambda x: tf.split(x,[1,1,1,1,1],axis=-1))(params_layer)
    log_eta = tf.keras.layers.Lambda(lambda x: x[0] + x[1]*x[5] + x[4]*x[6] + (x[2]/(x[7] - x[3])))([A,a,B,To,n, logMw, log_shear, T])
    model=tf.keras.models.Model(inputs=[fp_in, logMw_norm, log_shear_norm, T_norm, logMw, log_shear, T], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.01), loss='mse')
    return model

def create_ViscNN_phys(n_features):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw_norm = tf.keras.Input(shape=(1,))
    T_norm = tf.keras.Input(shape=(1,))
    Mw = tf.keras.Input(shape=(1,))
    shear = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, logMw_norm, T_norm])
    layer_1 = Dense(128, activation='relu', kernel_initializer='he_normal')(merged)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    params_layer = Dense(4, activation= None)(layer_2)
    log_k,a,tau,n = tf.keras.layers.Lambda(lambda x: tf.split(x,[1,1,1,1],axis=-1))(params_layer)
    zero_shear_visc = tf.keras.layers.Lambda(lambda x: x[0] + log10((x[2] ** x[1])))([log_k,a,Mw])
    log_eta = tf.keras.layers.Lambda(lambda x: x[0] - (x[1]-1)*log10(1 + (((10**x[0])*x[3])/x[2]))) ([zero_shear_visc, n, tau, shear])
    model = tf.keras.models.Model(inputs=[fp_in, logMw_norm, T_norm, Mw, shear], outputs=[zero_shear_visc])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.01), loss='mse')
    return model
