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
import torch
sys.path.append('../')
from Melt_Viscosity_Predictor.gpflow_tools.gpr_models import train_GPR
from Melt_Viscosity_Predictor.validation.metrics import OME


class history_obj:
    def __init__(self, hist):
        self.history = hist

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


def alpha_sig(x):
    return tf.keras.activations.sigmoid(x)*0.4 +3.2

def sig_mod(x):
    return tf.keras.activations.sigmoid(10*x)

def crit_sig(x):
    return tf.keras.activations.sigmoid(x) + 0.001

def Mcr_comp(M, Mcr):
    return tf.cast(tf.where(M<Mcr,0,1), tf.float32)

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

def create_ViscNN_concat(n_features, hp):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw = tf.keras.Input(shape=(1,))
    logMw_noise = GaussianNoise(0.02)(logMw)
    log_shear = tf.keras.Input(shape=(1,))
    shear_noise = GaussianNoise(0.02)(log_shear)
    T = tf.keras.Input(shape=(1,))
    T_noise = GaussianNoise(0.02)(T)
    P = tf.keras.Input(shape=(1,))
    fp_dense = Dense(hp['fp_dense'], activation = 'softplus', kernel_initializer='he_normal')(fp_in)
    merged = Concatenate(axis=1)([fp_dense, logMw_noise, shear_noise, T_noise, P])
    layer_1 = Dense(hp['layer_1'], activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(hp['l1_reg']))(merged)
    layer_1 = Dropout(hp['dropout_1'], input_shape = (hp['layer_1'],))(layer_1, training = True)
    layer_2 = Dense(hp['layer_2'], activation='softplus', kernel_initializer='he_normal', kernel_regularizer = l2(hp['l2_reg']))(layer_1)
    layer_2 = Dropout(hp['dropout_2'], input_shape = (hp['layer_2'],))(layer_2, training = True)
    log_eta = Dense(1, activation = activations.tanh)(layer_2)
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T, P], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model
  

def predict_all_cv(models, X_in, remove_model = None, is_torch = False, get_constants = False):
    """
    Gets mean and variance of predictions from all CV folds.
    """

    if not is_torch:
        # if remove_model:
        #     for r in remove_model: model_nums.pop(r)
        pred = []
        
        for m in models:
            #for layer in m.layers:
                #if isinstance(layer, tf.keras.layers.Dropout) and hasattr(layer, 'training'):
                 #   layer.training = True
            pred += [m(X_in, training = True)]
    else:
        X = [torch.tensor(x).to(models[0].device).float() for x in X_in]
        pred = []
        const_list = ['a1', 'a2', 'Mcr', 'kcr', 'c1', 'c2', 'Tr','tau', 'n', 'eta_0']
        constants = {k:[] for k in const_list}
        for m in models:
            for mod in m.modules():
                    if mod.__class__.__name__.startswith('Dropout'):
                        mod.train()
            if get_constants:
                y_pred, *const_out = m(*X, get_constants = True)
                y_pred = y_pred.cpu().detach().numpy()
                for k,i in zip(constants, range(len(const_out))):
                    constants[k] += [const_out[i].cpu().detach().numpy()]
            else:
                y_pred = m(*X).cpu().detach().numpy()
            
            #print('FOLD PREDICTION', y_pred)
            if not np.any(y_pred < -3):
                pred += [y_pred]
    
    pred = np.array(pred)
    std = np.std(pred, axis = 0)
    means = np.nanmean(pred, axis = 0)
    if get_constants:
        for k in constants:
            constants[k] = np.array(constants[k])
            constants[k] = np.nanmean(constants[k], axis = 0)
    try:
        if not get_constants:
            return [m[0] for m in means], [s[0] for s in std], pred
        else:
            return constants
    except:
        print(means)
        print(type(means))
        print('Error in prediction')

def load_models(date, data_type, NN_models = [create_ViscNN_concat]):
    NN = [[],[]]
    HypNet = []
    gpr = []
    history = [[],[]]
    for i in range(len(NN_models)):
        
        for n in range(10):
            try:
                NN[i].append(keras.models.load_model(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/model_{n}'))
                history[i].append(history_obj(np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/hist_{n}.npy', allow_pickle=True).item()))
            except FileNotFoundError:
                print(f'Could not find ANN files for {date}_{data_type} {i}...')
                continue
        NN_cv = np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/OME_CV.npy')
            
    
    
        for i in range(10):
            try:
                gpr.append(tf.saved_model.load(f'MODELS/{date}_{data_type}/GPR/model_{i}'))
            except FileNotFoundError:
                print(f'Could not find GPR files for {date}_{data_type} {i}...')
                continue

        gp_cv = np.load(f'MODELS/{date}_{data_type}/GPR/OME_CV.npy')

    for i in range(10):
        HypNet.append(torch.load(f'MODELS/{date}_{data_type}/PNN/model_{i}.pt'))
    HypNet_cv = np.load(f'MODELS/{date}_{data_type}/PNN/OME_CV.npy')

    return NN, history, gpr, gp_cv, NN_cv, HypNet, HypNet_cv

            
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