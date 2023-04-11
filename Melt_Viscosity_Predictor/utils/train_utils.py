from logging import raiseExceptions
from sklearn.model_selection import StratifiedKFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras_tuner as kt
from os import sys
sys.path.append('../')
from train_torch import run_training, MVDataset
from torch.utils.data.dataloader import DataLoader
import time
from math import floor
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
import torch
from .eval_utils import get_Mw_samples, get_shear_samples

def custom_train_test_split(data: pd.DataFrame(), test_id, id_col):
    test_df = data.loc[data[id_col].isin(test_id)]
    train_df = data.loc[~data[id_col].isin(test_id)]
    return train_df, test_df

def polymer_train_test_split(data: pd.DataFrame(), test_size: float, hold_out = 0):
    polymers = pd.read_excel('../Polymer-SMILES.xlsx')['Polymer']
    test_df = data.copy()
    it = 0
    if hold_out == 0:
        #print(len(test_df) / len(data))
        while (len(test_df) / len(data) >= test_size + 0.10 or len(test_df) / len(data) <= test_size - 0.10) and it < 20:
 
            train_poly, test_poly = train_test_split(polymers, test_size= test_size)
            test_poly.replace(r'[\W]','',inplace=True,regex=True)
            test_poly = test_poly.str.lower()
            data['Polymer'].replace(r'[\W]','',inplace=True,regex=True)
            data['Polymer'] = data['Polymer'].str.lower()
            test_poly_list = list(test_poly)
            if 'polyethene' in test_poly_list or 'linearhdpe' in test_poly_list or 'polyethylene' in test_poly_list:
                test_poly_list.append('linearhdpe')
                test_poly_list.append('polyethene')
                test_poly_list.append('polyethylene')
            if 'polypropylene' in test_poly_list or 'polyprop1ene' in test_poly_list:
                test_poly_list.append('polypropylene')
                test_poly_list.append('polyprop1ene')
            print(test_poly_list)
            
            poly_find = '|'.join(test_poly_list)
            data['Test'] = data['Polymer'].str.contains(poly_find)
            train_df = data[data['Test'] == False]
            test_df = data[data['Test'] == True]
            print(len(test_df) / len(data))
            it += 1
    else:
        test_poly = pd.Series([hold_out])
        test_poly.replace(r'[\W]','',inplace=True,regex=True)
        test_poly = test_poly.str.lower()
        print(test_poly)
        data['Polymer'].replace(r'[\W]','',inplace=True,regex=True)
        data['Polymer'] = data['Polymer'].str.lower()
        test_poly_list = list(test_poly)
        if 'polyethene' in test_poly_list or 'Linear HDPE' in test_poly_list or 'polyethylene' in test_poly_list:
            test_poly_list.append('linearhdpe')
            test_poly_list.append('polyethene')
            test_poly_list.append('polyethylene')
        if 'polypropylene' in test_poly_list or 'polyprop1ene' in test_poly_list:
            test_poly_list.append('polypropylene')
            test_poly_list.append('polyprop1ene')
        
        poly_find = '|'.join(test_poly_list)
        print('tp_list',test_poly_list)
        data['Test'] = data['Polymer'].str.contains(poly_find)
        train_df = data[data['Test'] == False]
        test_df = data[data['Test'] == True]

    return train_df, test_df

def assign_sample_ids(og_data: pd.DataFrame()):
    shear_samps, shear_ids = get_shear_samples(og_data.copy(), full = True)
    Mw_samps, mw_ids = get_Mw_samples(og_data.copy(), full = True);
    return pd.concat([shear_samps,Mw_samps], ignore_index = True), shear_ids, mw_ids

def hyperparam_opt(hypermodel, fit_in, eval_in, y_train, y_val, iter, train_type):
    model = hypermodel(fit_in[0].shape[1])

    tuner = kt.Hyperband(model, objective='val_loss',
                        max_epochs=30, hyperband_iterations = 1,
                        factor=3, project_name = f'ANN_{iter}', directory= f'HP_search_data/{datetime.date.today()}_{train_type}')

    tuner.search(fit_in, y_train,
                validation_data= (eval_in, y_val),
                epochs=30,
                batch_size = 20,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    
    return tuner.get_best_hyperparameters(1)[0]

def hyperparam_opt_ray(n_fp, train_loader, test_loader, device):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(5, 8)),
        "l2": tune.choice([0, 32, 64, 128]),
        "d1": tune.choice([0, 0.03,0.05, 0.1]),
        "d2": tune.choice([0, 0.03,0.05, 0.1]),
        "a_weight": tune.choice([0.001, 0.005, 0.01, 0.03, 0.05]),
        "weight_decay": tune.choice([0.00005, 0.0001, 0.0005, 0.001])
    }

    scheduler = HyperBandScheduler(max_t = 10)
    reporter = CLIReporter(metric_columns=["loss"])
    result = tune.run(
        partial(run_training, n_fp = n_fp, train_loader = train_loader, test_loader = test_loader, device = device, EPOCHS = 20, tuning = True),
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        metric = "loss",
        mode = "min",
        num_samples= 10,
        scheduler=scheduler,
        progress_reporter = reporter, 
        )

    print(result.default_metric)
    #print(result.)
    print([trial.metric_analysis.keys() for trial in result.trials])
    
    best = result.get_best_trial(metric = "loss", mode = "min", scope = "last")
    
    return best.config

def test_PIHN(XX, yy, M, S, T, P, scalers = None):
    M_scaler = scalers["M"]
    S_scaler = scalers["S"]
    y_scaler = scalers["y"]
    T_scaler = scalers["T"]
    S_torch_scaler = scalers["s_torch"]
    #TORCH NN TUNING
    #With scaling
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for train_index, test_index in kf.split(XX, np.where(S == 0,0,1)):
        X_train, X_val = XX[train_index], XX[test_index]
        y_train, y_val = yy[train_index], yy[test_index]
        M_train, M_val = M[train_index], M[test_index]
        S_train, S_val = S[train_index], S[test_index]
        T_train, T_val = T[train_index], T[test_index]
        P_train, P_val = P[train_index], P[test_index]
        n_features = X_train.shape[1]
  
        
        fit_in = [X_train, y_scaler.transform(M_scaler.inverse_transform(M_train)), y_scaler.transform(S_scaler.inverse_transform(S_train)), T_train, P_train]
        eval_in = [X_val, y_scaler.transform(M_scaler.inverse_transform(M_val)), y_scaler.transform(S_scaler.inverse_transform(S_val)), T_val, P_val]
        
        #fit_in = [X_train, y_scaler.transform(M_scaler.inverse_transform(M_train)), S_train, T_train, P_train]
        #eval_in = [X_val, y_scaler.transform(M_scaler.inverse_transform(M_val)), S_val, T_val, P_val]
        #fit_in = [X_train, M_train, S_train, T_train, P_train]
        #eval_in = [X_val, M_val, S_val, T_val, P_val]
        

        print(max(fit_in[1]), min(fit_in[1]))
        print(max(fit_in[2]), min(fit_in[2]))
        #Without scaling
        # fit_in = [X_train, (M_scaler.inverse_transform(M_train)), (S_scaler.inverse_transform(S_train)), T_scaler.inverse_transform(T_train), P_train]
        # eval_in = [X_val, (M_scaler.inverse_transform(M_val)), (S_scaler.inverse_transform(S_val)), T_scaler.inverse_transform(T_val), P_val]
        
        
        #fit_in = [X_train, y_scaler.transform(M_scaler.inverse_transform(M_train)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S_train)) - 0.00001), T_train, P_train]
        #eval_in = [X_val, y_scaler.transform(M_scaler.inverse_transform(M_val)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S_val)) - 0.00001), T_val, P_val]
        
        fit_ten = [torch.tensor(a).to(device).float() for a in fit_in]
        val_ten = [torch.tensor(a).to(device).float() for a in eval_in]
        tr_load = DataLoader(MVDataset(*fit_ten, torch.tensor(y_train).to(device).float()), batch_size = 20, shuffle = True)
        val_load = DataLoader(MVDataset(*val_ten, torch.tensor(y_val).to(device).float()), batch_size = 20, shuffle = True) 

        try:
            best_config = hyperparam_opt_ray(n_features, tr_load, val_load, device)
        except:
            raise Exception('PIHN error in training.')
        
        break
