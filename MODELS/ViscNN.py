import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate, Lambda
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
import sys
sys.path.append('../')
from Melt_Viscosity_Predictor.gpflow_tools.gpr_models import create_GPR


def Mcr_gpr_train(OG_fp, pca, M_scaler, scaler):
    me_data = pd.read_excel('Data/EntanglementMW_fp.xlsx')
    fp = me_data[[c for c in me_data.columns if 'fp' in c]]
    fp[[c for c in OG_fp if c not in fp]] = 0
    fp = fp.drop(columns = [c for c in fp if c not in OG_fp])
    y = M_scaler.transform(np.log10(np.array(me_data['Me']*2)).reshape(-1,1))
    X_me = scaler.transform((pca.transform(fp[OG_fp])))
    gpr_Mcr = create_GPR(X_me, y)
    return gpr_Mcr

def alpha_sig(x):
    return tf.keras.activations.sigmoid(x)*0.4 +3.2

def n_sig(x):
    return tf.keras.activations.sigmoid(x)*0.4 +0.2

def above_Mcr(M, fp_in = None, Mcr = None):
    if isinstance(Mcr, type(None)):
        gpr = load_me_gpr()
        Mcr, _ = gpr.predict_y(fp_in)
    above_Mcr = tf.math.greater(M, Mcr)
    return tf.where(above_Mcr, 1, 0)

def log10(x):
    numerator = K.log(x)
    denominator = K.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def OME_loss(yy, y_pred):
    y_pred = tf.math.pow(10.0, y_pred)
    yy = tf.math.pow(10.0, yy)
    div = tf.math.divide_no_nan(y_pred,yy)
    l = tf.math.reduce_prod(div) - 1
    return tf.math.abs(l)

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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model

def create_ViscNN_concat(n_features = 158):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw = tf.keras.Input(shape=(1,))
    log_shear = tf.keras.Input(shape=(1,))
    T = tf.keras.Input(shape=(1,))
    above_Mcr = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, logMw, log_shear, T])
    layer_1 = Dense(128, activation='relu', kernel_initializer='he_normal')(merged)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    log_eta = Dense(1, activation = None)(layer_2)
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T, above_Mcr], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
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
    eta = tf.keras.layers.Lambda(lambda x: x[0] + x[1]*x[5] + x[4]*x[6] + (x[2]/(x[7] - x[3])))([A,a,B,To,n, logMw_norm, log_shear_norm, T_norm])
    model=tf.keras.models.Model(inputs=[fp_in, logMw_norm, log_shear_norm, T_norm, logMw, log_shear, T], outputs=[eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss=OME_loss)
    return model

def create_ViscNN_phys(n_features):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw_norm = tf.keras.Input(shape=(1,))
    T_norm = tf.keras.Input(shape=(1,))
    shear = tf.keras.Input(shape=(1,))
    above_Mcr = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, T_norm, above_Mcr])
    layer_1 = Dense(32, activation='relu', kernel_initializer='he_normal')(merged)
    layer_2 = Dense(32, activation='relu', kernel_initializer='he_normal')(layer_1)
    log_k = Dense(1, activation= None, name= 'log_k')(layer_2)
    tau = Dense(1, activation= None, name= 'tau')(layer_2)
    #log_k, tau = Lambda(lambda x: tf.split(x,[1,1],axis=-1), name = 'params_split')(params_layer)
    alpha = Dense(1, activation = alpha_sig)(layer_2)
    alpha_mod = Lambda(lambda x: x[0]**x[1], name = 'alpha')([alpha, above_Mcr])
    alpha_sc = Dense(1, activation = None, name = 'alpha_scale')(alpha_mod)
    n = Dense(1, activation = n_sig, name = 'n')(layer_2)
    #n_sc = Dense(1, activation = None, name = 'n_scale')(n)
    zero_shear_visc = Lambda(lambda x: x[0] + (x[2]*(x[1])),name = 'zero_shear_visc') ([log_k,alpha_sc,logMw_norm])
    crit_shear = Lambda(lambda x: x[0]/x[1], name = 'alpha')([zero_shear_visc, tau])
    log_eta = Lambda(lambda x: x[0] - (x[1]-1)*log10(1 + (x[3]/(x[2])))) ([zero_shear_visc, n, tau, shear]) #x[0] - (x[1])*log10(1 + (((10**x[0])*x[3])/x[2]))
    model = tf.keras.models.Model(inputs=[fp_in, logMw_norm, shear, T_norm, above_Mcr], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model
