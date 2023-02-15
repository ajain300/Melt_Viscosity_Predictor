from logging import raiseExceptions
from sklearn.model_selection import KFold, StratifiedKFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
from validation.metrics import MSE, OME
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras_tuner as kt
import json
from os import sys
import os
sys.path.append('../')
from MODELS.ViscNN import predict_all_cv, ViscNN_concat_HP
from MODELS.target import TargetNet
from train_torch import run_training, MVDataset
from torch.utils.data.dataloader import DataLoader
import time
from math import floor
import yaml
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
import torch

def viscNN_LC(create_model, X_tot, Y_tot, logMw, log_shear, Temp):
    """
    Learnng curve test for general NN.
    """
    
    tr_sizes = list(np.linspace(0.10, 0.9, 9))
    train_error = []
    test_error = []
    for size in tr_sizes:
        print('Conducting test for train size ' + str(size))
        XX, X_test, yy, y_test, M, M_test, S, S_test, T, T_test = train_test_split(X_tot, Y_tot, logMw, log_shear, Temp, test_size= 1 - size)
        models, _ = crossval_NN(create_model, XX, yy, M, S, T, verbose = 0) 
        fit_in = [XX, M, S, T]
        eval_in = [X_test, M_test, S_test, T_test]
        pred_train, _, __ = predict_all_cv(models, fit_in)
        pred_test, _ , __= predict_all_cv(models, eval_in)
        train_error.append(OME(pred_train, yy))
        test_error.append(OME(pred_test, y_test))
        print('Train_error  = ' + str(train_error[-1]))
        print('Test_error  = ' + str(test_error[-1]))
        #train_error.append(model.evaluate(fit_in, yy, verbose = 0))
        #test_error.append(model.evaluate(eval_in, y_test, verbose = 0))

    plt.plot(tr_sizes, train_error, c = 'orange')
    plt.plot(tr_sizes, test_error, c= 'blue')
    plt.legend(['Train','Test'])

    return tr_sizes, train_error, test_error

    #To run learning curve test ex:
    #n, train, test = viscNN_LC(create_ViscNN_comb, X_tot, Y_tot, logMw, log_shear, Temp, S_scaler= S_scaler, T_scaler=T_scaler,M_scaler= M_scaler)
    # plt.plot(n, train)
    # plt.plot(n, test)
    # plt.legend(['Train','Test'])

def custom_train_test_split(data: pd.DataFrame(), test_id, id_col):
    test_df = data.loc[data[id_col].isin(test_id)]
    train_df = data.loc[~data[id_col].isin(test_id)]
    return train_df, test_df

def polymer_train_test_split(data: pd.DataFrame(), test_size: float, hold_out = 0):
    polymers = pd.read_excel('Polymer-SMILES.xlsx')['Polymer']
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

def crossval_NN(create_model, XX, yy, M, S, T, verbose = 1, random_state = None, epochs = 500):
    kf = KFold(n_splits=10, shuffle = True, random_state = random_state)
    m = []
    hist = []
    for train_index, test_index in kf.split(XX):
        X_train, X_val = XX[train_index], XX[test_index]
        y_train, y_val = yy[train_index], yy[test_index]
        M_train, M_val = M[train_index], M[test_index]
        S_train, S_val = S[train_index], S[test_index]
        T_train, T_val = T[train_index], T[test_index]
        n_features = X_train.shape[1]
        m.append(create_model(n_features))
        fit_in = [X_train, M_train, S_train, T_train]
        eval_in = [X_val, M_val, S_val, T_val]
        hist.append(m[-1].fit(fit_in, y_train, epochs=epochs, batch_size=30, validation_data = (eval_in, y_val) ,verbose=0))
        if verbose > 0:
            #print('MSE: %.3f, RMSE: %.3f' % (error[-1], np.sqrt(error[-1])))
            print('Trained fold ' + str(len(hist)) + ' ...')
            print('CV Error ' + str(len(hist)) + ': ' + str(hist[-1].history['val_loss'][-1]))
    #if verbose > 0:
        #print('CV MSE error:' + str(np.mean(error)))
    return m, hist

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


def custom_CV(NN_models, gpr_model, data):

    pass


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



def crossval_compare(NN_models, XX, yy, M, S, T, P, data = None, custom = False, verbose = 1, random_state = None, epochs = 500, gpr_model = None, save = True, train_type = 'split', scalers = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start_total = time.time()
    if custom:
        return custom_CV(NN_models, gpr_model, data)

    M_scaler = scalers["M"]
    S_scaler = scalers["S"]
    y_scaler = scalers["y"]
    T_scaler = scalers["T"]
    S_torch_scaler = scalers["s_torch"]
    M_torch_scaler = scalers["m_torch"]
    kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = random_state)
    NN = [[] for i in range(len(NN_models))]
    gpr = []
    PNN = []
    PNN_CV = []
    PNN_hist = []
    NN_cv = [[] for i in range(len(NN_models))]
    gp_cv = []
    hist = [[] for i in range(len(NN_models))]
    HPs = [[] for i in range(len(NN_models))]
    fold_num = 0
    for train_index, test_index in kf.split(XX, np.where(S == 0,0,1)):
        X_train, X_val = XX[train_index], XX[test_index]
        y_train, y_val = yy[train_index], yy[test_index]
        M_train, M_val = M[train_index], M[test_index]
        S_train, S_val = S[train_index], S[test_index]
        T_train, T_val = T[train_index], T[test_index]
        P_train, P_val = P[train_index], P[test_index]
        n_features = X_train.shape[1]
        fit_in = [X_train, M_train, S_train, T_train, P_train]
        eval_in = [X_val, M_val, S_val, T_val, P_val]
        
        #KERAS NN TUNING
        if True:
            for i in range(len(NN_models)): 
                #Hyperparam Tuning
                hyper_params = hyperparam_opt(ViscNN_concat_HP, fit_in, eval_in, y_train, y_val, fold_num, train_type)
                HPs[i].append(hyper_params.values)
                print(f'hyperparams',hyper_params.values)
                NN[i].append(NN_models[i](n_features, hyper_params)) #S_trans.inverse_transform(S_scaler.inverse_transform(S_train))
            for i in range(len(NN_models)): 
                s = time.time()
                hist[i].append(NN[i][-1].fit(fit_in, y_train, epochs=epochs, batch_size=20, validation_data = (eval_in, y_val) ,verbose=0))
                print(f'{NN_models[i].__name__} train time = {time.time() - s}')
                NN_cv[i].append(OME(y_val, NN[i][-1].predict(eval_in)))      
                plt.figure()
                plt.scatter(y_val, NN[i][-1].predict(eval_in))
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.plot([-1,1], [-1,1], 'k')
                os.makedirs(f'MODELS/{datetime.date.today()}_{train_type}/ANN', exist_ok = True)
                plt.savefig(f'MODELS/{datetime.date.today()}_{train_type}/ANN/cv_{fold_num}_parity.png')
            #TORCH NN TUNING
            #With scaling
            # fit_in = [X_train, M_train, S_train, T_train, P_train]
            # eval_in = [X_val, M_val, S_val, T_val, P_val]
            fit_in = [X_train, y_scaler.transform(M_scaler.inverse_transform(M_train)), y_scaler.transform(S_scaler.inverse_transform(S_train)), T_train, P_train]
            eval_in = [X_val, y_scaler.transform(M_scaler.inverse_transform(M_val)), y_scaler.transform(S_scaler.inverse_transform(S_val)), T_val, P_val]
            

            print(max(fit_in[1]), min(fit_in[2]), max(fit_in[3]))
            #Without log scaling
            # fit_in = [X_train, M_torch_scaler.transform(np.power(10, (M_scaler.inverse_transform(M_train)))), (S_scaler.inverse_transform(S_train)), T_scaler.inverse_transform(T_train), P_train]
            # eval_in = [X_val, M_torch_scaler.transform(np.power(10, (M_scaler.inverse_transform(M_val)))), (S_scaler.inverse_transform(S_val)), T_scaler.inverse_transform(T_val), P_val]
            
            
            # fit_in = [X_train, y_scaler.transform(M_scaler.inverse_transform(M_train)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S_train)) - 0.00001), T_train, P_train]
            # eval_in = [X_val, y_scaler.transform(M_scaler.inverse_transform(M_val)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S_val)) - 0.00001), T_val, P_val]
            
            fit_ten = [torch.tensor(a).to(device).float() for a in fit_in]
            val_ten = [torch.tensor(a).to(device).float() for a in eval_in]
            tr_load = DataLoader(MVDataset(*fit_ten, torch.tensor(y_train).to(device).float()), batch_size = 20, shuffle = True)
            val_load = DataLoader(MVDataset(*val_ten, torch.tensor(y_val).to(device).float()), batch_size = 20, shuffle = True) 

            best_config = hyperparam_opt_ray(n_features, tr_load, val_load, device)
            print('Best Configuration from RayTune ', best_config)
            model, train_loss, val_loss = run_training(best_config, n_fp = n_features,train_loader = tr_load, test_loader = val_load, device = device, EPOCHS = 200)
            PNN.append(model)
            PNN_hist.append(val_loss)
            val_pred = model(*val_ten).cpu().detach().numpy()
            print(type(model))
            PNN_CV.append(OME(y_val, val_pred))
            plt.figure()
            plt.scatter(y_val, val_pred)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot([-1,1], [-1,1], 'k')
            os.makedirs(f'MODELS/{datetime.date.today()}_{train_type}/PNN', exist_ok = True)
            plt.savefig(f'MODELS/{datetime.date.today()}_{train_type}/PNN/cv_{fold_num}_parity.png')

            fit_in = [X_train, M_train, S_train, T_train, P_train]
            eval_in = [X_val, M_val, S_val, T_val, P_val]
        #GPR MODEL TRAINING
        if gpr_model:
            X_train_ = np.concatenate((X_train, M_train, S_train, T_train, P_train), axis = 1)
            X_test_ = np.concatenate((X_val, M_val, S_val, T_val, P_val), axis = 1)
            s = time.time()
            gpr.append(gpr_model(X_train_, y_train))
            print(f'GPR train time = {time.time() - s}')
            m = gpr[-1]
            mean, var = m.predict_y(X_test_)
            gp_cv.append(OME(y_val, mean))
            plt.figure()
            plt.scatter(y_val, mean)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot([-1,1], [-1,1], 'k')
            os.makedirs(f'MODELS/{datetime.date.today()}_{train_type}/GPR', exist_ok = True)
            plt.savefig(f'MODELS/{datetime.date.today()}_{train_type}/GPR/cv_{fold_num}_parity.png')

        if verbose > 0:
            #print('MSE: %.3f, RMSE: %.3f' % (error[-1], np.sqrt(error[-1])))
            print('Trained fold ' + str(len(hist[-1])) + ' ...')
            for i in range(len(NN_models)):
                print('MSE CV Error ' + NN_models[i].__name__ + ': ' + str(hist[i][-1].history['val_loss'][-1]))
                print('OME CV Error ' + NN_models[i].__name__ + ': ' + str(NN_cv[i][-1]))
            if gpr_model: print('OME CV Error GPR: ' + str(gp_cv[-1]))
        fold_num += 1
    #if verbose > 0:
        #print('CV MSE error:' + str(np.mean(error)))
    print(f'Overall CV time = {time.time() - start_total}')
    
    #save models
    if save:
        print('Saving models...')
        for i in range(len(NN_models)):
            for n in range(len(NN[i])):
                NN[i][n].save(f'MODELS/{datetime.date.today()}_{train_type}/{NN_models[i].__name__}/model_{n}')
                np.save(f'MODELS/{datetime.date.today()}_{train_type}/{NN_models[i].__name__}/hist_{n}',hist[i][n].history)
                try:
                    with open(f'MODELS/{datetime.date.today()}_{train_type}/{NN_models[i].__name__}/model_{n}_hp.json', 'w') as fp:
                        json.dump(HPs[i][n], fp)
                    fp.close()
                except:
                    print('Could not print to json.')
            print(f'Saved ANN_{i}')
            np.save(f'MODELS/{datetime.date.today()}_{train_type}/{NN_models[i].__name__}/OME_CV', np.array(NN_cv[i]))
        for i in range(len(PNN)):
            torch.save(PNN[i], f'MODELS/{datetime.date.today()}_{train_type}/PNN/model_{i}.pt')
            np.save(f'MODELS/{datetime.date.today()}_{train_type}/PNN/OME_CV', np.array(gp_cv))
            print(f'Saved PNN_{i}')
        for i in range(len(gpr)):
            gpr[i].predict_f_compiled = tf.function(gpr[i].predict_f, input_signature=[tf.TensorSpec(shape=[None, X_train_.shape[1]], dtype=tf.float64)])
            tf.saved_model.save(gpr[i], f'MODELS/{datetime.date.today()}_{train_type}/GPR/model_{i}')
            print(f'Saved GPR_{i}')
        np.save(f'MODELS/{datetime.date.today()}_{train_type}/GPR/OME_CV', np.array(gp_cv))
        print('saved models')
    return NN, hist, gpr, gp_cv, NN_cv, PNN, PNN_CV, PNN_hist

def delete_outlier(model, y_pred, y_test, TH = 12):
    ind = []
    for i in range(y_pred.shape[0]):
        if y_pred[i] > TH:
            y_pred = np.delete(y_pred, i, 0)
            y_test = np.delete(y_test, i, 0)
            ind.append(i)
            break

    return y_pred, y_test, ind

def get_Mw_samples(data:pd.DataFrame, full = False):
    id = 1000
    fp_cols = []
    data = data.loc[data['Shear_Rate'] == 0.0]
    #data['Shear_Rate'] = 0
    data = data.sort_values(['Polymer', 'Temperature'])
    data = data.reset_index(drop = True)
    if len(data) == 0:
        print('data_len = 0',data)
        return [], []
    for c in data.columns:
        if 'fp' in c:
            fp_cols.append(c)
    
    temp = data.loc[0, 'Temperature']
    poly = data.loc[0, 'Polymer']
    fp = data.loc[0, fp_cols]
    
    for i in data.index:
        if data.loc[i,'Temperature'] == temp and data.loc[i, 'Polymer'] == poly and data.loc[i, fp_cols].equals(fp):
            data.loc[i,'SAMPLE_ID'] = id
        else:
            id += 1
            data.loc[i, 'SAMPLE_ID'] = id

        temp = data.loc[i, 'Temperature']
        poly = data.loc[i, 'Polymer']
        fp = data.loc[i, fp_cols]

    
    sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])

    if not full:
        for c in ['Mw', 'Melt_Viscosity']:
            data[c] = np.log10(data[c])
        for i in sample_id:
            if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
                #print()
                data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
    sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])
    return data, sample_id

    
def Mw_extrapolation(data:pd.DataFrame, n_samples):
    """
    Takes in overall data and extrapolates zero-shear Mw values from samples
    """
    id = 0
    temp = data.loc[0, 'Temperature']
    poly = data.loc[0, 'Polymer']
    for i in data.index:
        if data.loc[i, 'Temperature'] == temp and data.loc[i, 'Polymer'] == poly:
            data.loc[i, 'SAMPLE_ID'] = id
        else:
            id += 1
            data.loc[i, 'SAMPLE_ID'] = id

        temp = data.loc[i, 'Temperature']
        poly = data.loc[i, 'Polymer']

    fp_cols = []
    for c in data.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            fp_cols.append(c)

    for c in ['Mw', 'Melt_Viscosity']:
        data[c] = np.log10(data[c])

    z_shear_samples = []
    sample_id = list(data.loc[data['Shear_Rate'] == 0].agg({'SAMPLE_ID': 'unique'})[0])

    for i in range(len(sample_id)):
        if sum(data.loc[data['SAMPLE_ID'] == sample_id[i], 'Shear_Rate'].to_list()) == 0:
            z_shear_samples.append(i)
    PL_data = pd.DataFrame(columns = ['SAMPLE_ID', 'Polymer', 'SMILES', 'Temperature', 'Mcr', 'Log_K1', 'Log_K2', 'Alpha'])
    data = data[data['Shear_Rate'] == 0]
    print('Found Zero Shear Samples')
    sample_data = []
    for i in z_shear_samples:
        x = data.loc[data['SAMPLE_ID'] == i, 'Mw'].to_list()
        y = data.loc[data['SAMPLE_ID'] == i, 'Melt_Viscosity'].to_list()
        T =  data.loc[data['SAMPLE_ID'] == i, 'Temperature'].to_list()
        if len(x) < 5: continue
        if type(x[0]) != float or pd.isnull(x[0]): continue

        ind = np.argsort(x)
        x = [x[i] for i in ind]
        y = [y[i] for i in ind]
        #print(i)
        sample_data.append([x,y])
        poly = data.loc[data['SAMPLE_ID'] == i, 'Polymer'].to_list()[0]
        temp = T[0]
        p = 2
        Rel_E = 0
        log_k_old = 0
        while Rel_E < 0.1:
            #print('p = ' + str(p))
            log_k1 = np.mean(np.array(y[0:p]) - np.array(x[0:p]))
            #print('logk1 = ' + str(log_k1))

            y_pred = (x[0:p]) + log_k1

            RMSE = np.sqrt(np.sum(np.power(y[0:p] - y_pred,2))/len(y_pred))
            Rel_E = RMSE
            #print("Rel_E = " + str(Rel_E))
            p+=1
            log_k_old = log_k1

        p -= 2
        log_k1 = log_k_old
        if p == 1 or p == 0:
            p = 0
            log_k1 = 0
        #print("p final = " + str(p))

        x2 = x[p:]
        y2 = y[p:]

        fit = np.polyfit(x2,y2,1)
        if p > 0: Mcr = np.power(10, (log_k1 - fit[1])/ (fit[0] - 1))
        #print(fit)

        y1_pred = np.array(x[0:p]) + log_k1
        y2_pred = np.array(x2)*fit[0] + fit[1]

        if np.sqrt(np.sum(np.power(y2 - y2_pred,2))/len(y2_pred)) < 0.5:
            PL_data = PL_data.append({'SAMPLE_ID' : i, 'Polymer': poly,
                    'SMILES' : data.loc[data['SAMPLE_ID'] == i, 'SMILES'].to_list()[0],
                    'Temperature' : temp,
                    'Mcr': Mcr if p > 0 else np.NaN, 'Log_K1': log_k1, 'Log_K2' : fit[1],
                    'Alpha': fit[0]}, ignore_index = True)
    #filter out unrealistic predictions
    print('Found LOBF for samples')
    PL_data = PL_data.merge(data[fp_cols + ['SMILES']], on = 'SMILES', how = 'left').drop_duplicates()
    #PL_data = PL_data.groupby('SAMPLE_ID').max().reset_index()
    PL_data = PL_data[[x == y for x, y in zip(PL_data['Alpha'] > 2, PL_data['Alpha'] < 5)]]

    out = pd.DataFrame()
    y_pred = []
    M = []
    for i in np.random.choice(PL_data.index, size= n_samples):
        print('Generating sample...' + str(len(M)))
        log_Mw = np.random.rand(1)*3 + 4 #if PL_data.loc[i, 'Log_K2'] != 0 else np.random.rand(1)*4 + 3
        if  PL_data.loc[i, 'Log_K2'] != 0:
            y_pred.append(log_Mw*PL_data.loc[i, 'Alpha'] +  PL_data.loc[i, 'Log_K2'])
        else:
            if np.log10(PL_data.loc[i, 'Mcr']) > log_Mw:
                y_pred.append(log_Mw*PL_data.loc[i, 'Alpha'] +  PL_data.loc[i, 'Log_K2'])
            else:
                y_pred.append(log_Mw + PL_data.loc[i, 'Log_K1'])
        out = out.append(PL_data.loc[i]).reset_index(drop = True)
        y_pred[-1] = y_pred[-1][0]
        M.append(log_Mw[0])

    out['Melt_Viscosity'] = np.power(10, y_pred)
    out['Mw'] = np.power(10, M)
    PL_data['Shear_Rate'] = 0
    out['Shear_Rate'] = 0
    return out, PL_data.reset_index(drop = True), sample_data




def Mw_test(samples_df: pd.DataFrame, samp):
    fp_cols = []
    for c in samples_df.columns:
        if 'fp' in c:
            fp_cols.append(c)
    logMw = pd.Series(np.linspace(1,7,40))
    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)

    fp = trial.loc[0,fp_cols] #fp = trial.loc[0,fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['logMw'] = logMw
    tests['Temperature'] = trial['Temperature'].values[0]
    tests['Shear_Rate'] = trial['Shear_Rate'].values[0]
    tests['Polymer'] = trial['Polymer'].values[0]
    tests['PDI'] = trial['PDI'].values[0]
    tests.loc[[i for i in tests.index], fp_cols] = np.array(fp)
    #tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)

    if trial['Shear_Rate'].values[0] != 0:
        tests['Shear_Rate'] = np.log10(tests['Shear_Rate'])
        tests['SHEAR'] = 1
        tests['ZERO_SHEAR'] = 0
    else:
        tests['SHEAR'] = 0
        tests['ZERO_SHEAR'] = 1

    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    P = np.array(tests['PDI']).reshape(-1, 1)
    OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    # if trial['Log_K1'].values[0] != 0:
    #     tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'logMw'] + trial['Log_K1'].values[0]
    #     tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
    # else:
    #     tests['Melt_Viscosity'] = tests['logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
    p = 'Polymer'
    t = 'Temperature'
    Mw_exp = trial['Mw']
    P_exp = trial['PDI']
    visc_exp = trial['Melt_Viscosity']
    out = {'exp': [Mw_exp, visc_exp, P_exp], 'data_in':[XX, OH_shear,M,S,T, P], 'sample': f'{trial[p].values[0]} at {trial[t].values[0]} C'}

    return out

def get_shear_samples(data:pd.DataFrame, full = False):
    """
    Takes in overall data and extrapolates zero-shear Mw values from samples
    """
    fp_cols = []
    data = data.reset_index(drop = True)
    for c in data.columns:
        if 'fp' in c:
            fp_cols.append(c)
    id = 2000
    temp = data.loc[data.index[0], 'Temperature']
    poly = data.loc[data.index[0], 'Polymer']
    weight = data.loc[data.index[0], 'Mw']
    fp = data.loc[data.index[0], fp_cols]

    if len(data) == 0:
        return [], []
    for i in data.index:
        if data.loc[i, 'Temperature'] == temp and data.loc[i, 'Polymer'] == poly and weight == data.loc[i, 'Mw'] and data.loc[i, fp_cols].equals(fp):
            data.loc[i, 'SAMPLE_ID'] = id
        else:
            id += 1
            data.loc[i, 'SAMPLE_ID'] = id
        temp = data.loc[i, 'Temperature']
        poly = data.loc[i, 'Polymer']
        weight = data.loc[i, 'Mw']
        fp = data.loc[i, fp_cols]

    sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])
    for samp in sample_id:
        if sum(data.loc[data['SAMPLE_ID'] == samp, 'Shear_Rate']) == 0:
            data = data.drop(data.loc[data['SAMPLE_ID'] == samp].index, axis = 0)

    sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])
    
        
    if not full:
        for c in ['Mw']:
            data[c] = np.log10(data[c])
        for i in sample_id:
            if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
                data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
        sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])

    return data, sample_id


def shear_test(samples_df: pd.DataFrame, samp):
    out = []
    fp_cols = []
    samples_df = samples_df.loc[samples_df['Shear_Rate'] != 0]
    for c in samples_df.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            fp_cols.append(c)
    log_shear = pd.Series(np.linspace(-5,6,50))
    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)
    fp = trial.loc[0, fp_cols]
    #fp = trial.loc[0, fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['Shear_Rate'] = np.power(10, log_shear)
    tests['logMw'] = trial.loc[0,'Mw']
    tests['Temperature'] = trial.loc[0,'Temperature']
    tests['PDI'] = trial.loc[0,'PDI']
    tests.loc[[i for i in tests.index], fp_cols] = np.array(fp)
    #tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)

    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    P = np.array(tests['PDI']).reshape(-1, 1)
    #OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    p = 'Polymer'
    t = 'Temperature'
    m = 'Mw'
    y = np.array(trial['Melt_Viscosity']).reshape(-1, 1)
    trial_shear = np.log10(np.array(trial['Shear_Rate'])).reshape(-1,1)
    P_exp = np.array(trial['PDI']).reshape(-1, 1)
    out = {'exp':[trial_shear, y, P_exp], 'data_in':[XX,M,S,T,P], 'sample': f'{trial[p].values[0]} at {trial[t].values[0]} C, Mw = {floor(np.power(10, trial.loc[0,m]))}'}
    return out

def get_temp_samples(data:pd.DataFrame, full = False):
    """
    Takes in overall data and extrapolates temperature values from samples
    """
    fp_cols = []
    data = data.reset_index(drop = True)
    for c in data.columns:
        if 'fp' in c:
            fp_cols.append(c)
    id = 3000
    if len(data) == 0:
        return [], []
    
    shear = data.loc[data.index[0], 'Shear_Rate']
    poly = data.loc[data.index[0], 'Polymer']
    weight = data.loc[data.index[0], 'Mw']
    fp = data.loc[data.index[0], fp_cols]


    for i in data.index:
        if data.loc[i, 'Shear_Rate'] == shear and data.loc[i, 'Polymer'] == poly and weight == data.loc[i, 'Mw'] and data.loc[i, fp_cols].equals(fp):
            data.loc[i, 'SAMPLE_ID'] = id
        else:
            id += 1
            data.loc[i, 'SAMPLE_ID'] = id
        shear = data.loc[i, 'Shear_Rate']
        poly = data.loc[i, 'Polymer']
        weight = data.loc[i, 'Mw']
        fp = data.loc[i, fp_cols]

    # sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])
    # for samp in sample_id:
    #     if sum(data.loc[data['SAMPLE_ID'] == samp, 'Shear_Rate']) == 0:
    #         data = data.drop(data.loc[data['SAMPLE_ID'] == samp].index, axis = 0)

    sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])
    
        
    if not full:
        for c in ['Mw']:
            data[c] = np.log10(data[c])
        for i in sample_id:
            if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
                data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
        sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])

    return data, sample_id




def temp_test(samples_df: pd.DataFrame, samp, temp_range = 50):
    fp_cols = []
    for c in samples_df.columns:
        if 'fp' in c:
            fp_cols.append(c)


    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)
    print(trial)
    temps = pd.Series(np.linspace(trial['Temperature'].values[0] - temp_range,trial['Temperature'].values[-1] + temp_range,40))
    print(temps)


    fp = trial.loc[0,fp_cols] #fp = trial.loc[0,fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['Temperature'] = temps
    tests['logMw'] = trial['Mw'].values[0]
    tests['Shear_Rate'] = trial['Shear_Rate'].values[0]
    tests['Polymer'] = trial['Polymer'].values[0]
    tests['PDI'] = trial['PDI'].values[0]
    tests.loc[[i for i in tests.index], fp_cols] = np.array(fp)
    #tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)

    if trial['Shear_Rate'].values[0] != 0:
        tests['Shear_Rate'] = np.log10(tests['Shear_Rate'])
        tests['SHEAR'] = 1
        tests['ZERO_SHEAR'] = 0
    else:
        tests['SHEAR'] = 0
        tests['ZERO_SHEAR'] = 1

    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    P = np.array(tests['PDI']).reshape(-1, 1)
    OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    # if trial['Log_K1'].values[0] != 0:
    #     tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'logMw'] + trial['Log_K1'].values[0]
    #     tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
    # else:
    #     tests['Melt_Viscosity'] = tests['logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
    p = 'Polymer'
    t = 'Temperature'
    temp_exp = trial['Temperature']
    P_exp = trial['PDI']
    visc_exp = trial['Melt_Viscosity']
    out = {'exp': [temp_exp, visc_exp, P_exp], 'data_in':[XX,M,S,T, P], 'sample': f'{trial[p].values[0]}'}

    return out


def small_shear_test(samples_df: pd.DataFrame, samp):
    out = []
    fp_cols = []
    
    for c in samples_df.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            fp_cols.append(c)
    shear = pd.Series(np.linspace(0,1,30))

    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)
    fp = trial.loc[0, fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['Shear_Rate'] = shear
    tests['logMw'] = trial.loc[0,'Mw']
    tests['Temperature'] = trial.loc[0,'Temperature']
    tests['PDI'] = trial.loc[0,'PDI']
    tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)


    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    P = np.array(tests['PDI']).reshape(-1, 1)
    #OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    p = 'Polymer'
    t = 'Temperature'
    m = 'Mw'
    y = np.array(trial['Melt_Viscosity']).reshape(-1, 1)
    trial_shear = np.array(trial['Shear_Rate']).reshape(-1,1)
    #print(trial)
    out = {'known':[trial_shear, y], 'data_in':[XX,M,S,T,P], 'sample': f'{trial[p].values[0]} at {trial[t].values[0]} C, Mw = {floor(np.power(10, trial.loc[0,m]))}'}
    return out

def evaluate_model(Y_test, Y_train, test_df, train_df):
    """
    Get model predictions and see what samples are predicted badly.
    """
    test_df['Y_pred'] = Y_test
    train_df['Y_pred'] = Y_train

    test_df['Error'] = abs(test_df['Melt_Viscosity'] - test_df['Y_pred'])#/test_df['Melt_Viscosity']
    train_df['Error'] = abs(train_df['Melt_Viscosity'] - train_df['Y_pred'])#/train_df['Melt_Viscosity']

    test_std = np.std(test_df['Error'])
    train_std = np.std(train_df['Error'])

    print(train_std)

    test_df['BAD_PRED'] = test_std < test_df['Error']
    train_df['BAD_PRED'] = train_std < train_df['Error']
    
    return test_df, train_df