import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Dropout, Activation, GaussianNoise
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
import kerastuner as kt
import sys
import matplotlib.pyplot as plt
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
        merged = Concatenate(axis=1)([fp_in, logMw, log_shear, T])
        L1_size = hp.Int('layer_1', 30, 150, step = 30)
        L2_size = hp.Int('layer_2', 30, 150, step = 30)
        #L3_size = hp.Int('layer_3', 30, 210, step = 30)
        layer_1 = Dense(L1_size, activation='softplus', kernel_initializer='he_normal')(merged)
        layer_1 = Dropout(hp.Float('dropout_1', 0, 0.5, step = 0.05), input_shape = (L1_size,))(layer_1)
        layer_2 = Dense(L2_size, activation='softplus', kernel_initializer='he_normal')(layer_1)
        layer_2 = Dropout(hp.Float('dropout_2', 0, 0.5, step = 0.05), input_shape = (L2_size,))(layer_2)
        #layer_3 = Dense(L3_size, activation='softplus', kernel_initializer='he_normal')(layer_2)
        #layer_3 = Dropout(hp.Float('dropout_3', 0, 0.4, step = 0.1), input_shape = (L3_size,))(layer_3)
        log_eta = Dense(1, activation = None)(layer_2)
        model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T], outputs=[log_eta])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
        return model

def Mcr_gpr_train(OG_fp, pca, M_scaler, scaler, transform = True):
    me_data = pd.read_excel('Data/EntanglementMW_fp.xlsx')
    fp = me_data[[c for c in me_data.columns if 'fp' in c]]
    fp[[c for c in OG_fp if c not in fp]] = 0
    fp = fp.drop(columns = [c for c in fp if c not in OG_fp])
    y = M_scaler.transform(np.log10(np.array(me_data['Me']*2)).reshape(-1,1))
    X_me = scaler.transform((pca.transform(fp[OG_fp]))) if transform else  scaler.transform(fp[OG_fp])
    gpr_Mcr, cv_error = train_GPR(X_me, y, LOO = True)
    pred, var = gpr_Mcr.predict_y(X_me)
    plt.plot(np.linspace((min(y)[0]), (max(y)[0]), num = 2),np.linspace((min(y)[0]), (max(y)[0]), num = 2),'k-')
    plt.ylabel('Scaled Mw Prediction')
    plt.xlabel('Scaled Mw Truth')
    plt.title('Mcr Prediction')
    plt.scatter(y,pred, c = 'orange')
    return gpr_Mcr, cv_error

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
    merged = Concatenate(axis=1)([fp_in, logMw_noise, shear_noise, T_noise])
    layer_1 = Dense(hp['layer_1'], activation='softplus', kernel_initializer='he_normal')(merged)
    layer_1 = Dropout(hp['dropout_1'], input_shape = (hp['layer_1'],))(layer_1)
    layer_2 = Dense(hp['layer_2'], activation='softplus', kernel_initializer='he_normal')(layer_1)
    layer_2 = Dropout(hp['dropout_2'], input_shape = (hp['layer_2'],))(layer_2)
    log_eta = Dense(1, activation = 'softplus')(layer_2)
    model=tf.keras.models.Model(inputs=[fp_in, logMw, log_shear, T], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model

def create_ViscNN_phys_HP(hp):
    fp_in = tf.keras.Input(shape=(217,))
    logMw_norm = tf.keras.Input(shape=(1,))
    T_norm = tf.keras.Input(shape=(1,))
    shear = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, T_norm])
    L1_size = hp.Int('layer_1', 30, 210, step = 30)
    L2_size = hp.Int('layer_2', 30, 210, step = 30)
    #L3_size = hp.Int('layer_3', 30, 210, step = 30)
    layer_1 = Dense(L1_size, activation='softplus', kernel_initializer='he_normal')(merged)
    layer_1 = Dropout(hp.Float('dropout_1', 0, 0.4, step = 0.1), input_shape = (L1_size,))(layer_1)
    layer_2 = Dense(L2_size, activation='softplus', kernel_initializer='he_normal')(layer_1)
    layer_2 = Dropout(hp.Float('dropout_2', 0, 0.4, step = 0.1), input_shape = (L2_size,))(layer_2)
    log_k_1 = Dense(1, activation= None, name= 'log_k_1')(layer_2)
    log_k_2 = Dense(1, activation = 'softplus' , name = 'log_k_2')(layer_2)
    #log_k = Dense(1, activation= None, name= 'log_k')(Concatenate(axis=1)([log_k, ab]))
    tau = Dense(1, activation= 'softplus', name= 'tau')(layer_2)
    alpha_1 = Dense(1, activation = 'softplus' , name = 'alpha_1')(layer_2)
    alpha_2 = Dense(1, activation = 'softplus' , name = 'alpha_2')(layer_2)
    Mcr_calc = hp.Choice('Mcr_calc', [1,2,3])
    opt = {1: fp_in, 2: layer_1, 3:layer_2}
    Mcr = Dense(1, activation = 'softplus' , name = 'Mcr')(opt[Mcr_calc])
    decision_layer = Concatenate(axis=1)([alpha_1, alpha_2, log_k_1, log_k_2, Mcr, logMw_norm])
    dec_layer = hp.Int('dec_layer', 2, 6, step = 1)
    decision_layer = Dense(dec_layer, activation = 'softplus' , name = 'dec_layer')(decision_layer)
    alpha = Dense(1, activation = 'softplus' , name = 'alpha')(decision_layer)
    log_k = Dense(1, activation = 'softplus' , name = 'log_k')(decision_layer)
    n = Dense(1, activation = 'softplus', name = 'n')(layer_2)
    #n_sc = Dense(1, activation = None, name = 'n_scale')(n)
    zero_shear_visc = Lambda(lambda x: x[0] + (x[2]*(x[1])),name = 'zero_shear_visc') ([log_k,alpha,logMw_norm])
    crit_shear = Dense(1, activation = 'softplus' , name = 'crit_shear')(Concatenate(axis=1)([zero_shear_visc, tau]))
    #crit_shear_sc = Dense(1, activation = crit_sig)(crit_shear)
    log_eta = Lambda(lambda x: x[0] - (x[1])*log10(1 + (x[3]/x[2]))) ([zero_shear_visc, n, crit_shear, shear]) #x[0] - (x[1])*log10(1 + (((10**x[0])*x[3])/x[2]))
    model = tf.keras.models.Model(inputs=[fp_in, logMw_norm, shear, T_norm], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss=OME)
    return model


def create_ViscNN_phys(n_features):
    fp_in = tf.keras.Input(shape=(n_features,))
    logMw_norm = tf.keras.Input(shape=(1,))
    T_norm = tf.keras.Input(shape=(1,))
    shear = tf.keras.Input(shape=(1,))
    merged = Concatenate(axis=1)([fp_in, T_norm])
    layer_1 = Dense(150, activation='softplus', kernel_initializer='he_normal')(merged)
    layer_1 = Dropout(0.1, input_shape = (210,))(layer_1)
    layer_2 = Dense(90, activation='softplus', kernel_initializer='he_normal')(layer_1)
    layer_2 = Dropout(0.1, input_shape = (90,))(layer_2)
    log_k_1 = Dense(1, activation= None, name= 'log_k_1')(layer_2)
    log_k_2 = Dense(1, activation = 'softplus' , name = 'log_k_2')(layer_2)
    #log_k = Dense(1, activation= None, name= 'log_k')(Concatenate(axis=1)([log_k, ab]))
    tau = Dense(1, activation= 'softplus', name= 'tau')(layer_2)
    Mcr = Dense(1, activation = 'softplus' , name = 'Mcr')(fp_in)
    alpha_1 = Dense(1, activation = 'softplus' , name = 'alpha_1')(layer_2)
    alpha_2 = Dense(1, activation = 'softplus' , name = 'alpha_2')(layer_2)
    above_Mcr =  Lambda(lambda x: 10*(x[0] - x[1]))([logMw_norm, Mcr])
    above_Mcr = Activation('sigmoid')(above_Mcr)
    decision_layer = Concatenate(axis=1)([alpha_1, alpha_2, log_k_1, log_k_2, above_Mcr])
    decision_layer = Dense(5, activation = 'softplus' , name = 'dec_layer')(decision_layer)
    alpha = Dense(1, activation = 'softplus' , name = 'alpha')(decision_layer)
    log_k = Dense(1, activation = 'softplus' , name = 'log_k')(decision_layer)
    n = Dense(1, activation = 'softplus', name = 'n')(layer_2)
    #n_sc = Dense(1, activation = None, name = 'n_scale')(n)
    zero_shear_visc = Lambda(lambda x: x[0] + (x[2]*(x[1])),name = 'zero_shear_visc') ([log_k,alpha,logMw_norm])
    crit_shear = Dense(1, activation = 'softplus' , name = 'crit_shear')(Concatenate(axis=1)([zero_shear_visc, tau]))
    #crit_shear_sc = Dense(1, activation = crit_sig)(crit_shear)
    log_eta = Lambda(lambda x: x[0] - (x[1])*log10(1 + (x[3]/x[2]))) ([zero_shear_visc, n, crit_shear, shear]) #x[0] - (x[1])*log10(1 + (((10**x[0])*x[3])/x[2]))
    model = tf.keras.models.Model(inputs=[fp_in, logMw_norm, shear, T_norm], outputs=[log_eta])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
    return model

def predict_all_cv(models, X_in, remove_model = None):
    """
    Gets mean and variance of predictions from all CV folds.
    """
    if remove_model:
        for r in remove_model: model_nums.pop(r)
    pred = []
    for m in models:
        pred += [m.predict(X_in)]
    pred = np.array(pred)
    std = np.std(pred, axis = 0)
    means = np.nanmean(pred, axis = 0)

    return [m[0] for m in means], [s[0] for s in std], pred

def load_models(date, data_type, NN_models = [create_ViscNN_concat, create_ViscNN_phys]):
    NN = [[],[]]
    gpr = []
    history = [[],[]]
    for i in range(len(NN_models)):
        for n in range(10):
            NN[i].append(keras.models.load_model(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/model_{n}'))
            history[i].append(history_obj(np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/hist_{n}.npy', allow_pickle=True).item()))
        NN_cv = np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/OME_CV.npy')
    for i in range(10):
        gpr.append(tf.saved_model.load(f'MODELS/{date}_{data_type}/GPR/model_{i}'))
    gp_cv = np.load(f'MODELS/{date}_{data_type}/GPR/OME_CV.npy')
    return NN, history, gpr, gp_cv, NN_cv

            
class metalearner(tf.keras.Model):
    def __init__(CV_models):
        self.models

    def build():
        #for 
        
        predictions = tf.keras.Input(shape=(10,))
        HL = Dense(10, activation='softplus', kernel_initializer='he_normal')(predictions)
        model=tf.keras.models.Model(inputs=[predictions], outputs=[log_eta])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.001), loss='mse')
        return model