import tensorflow as tf
import pandas as pd
import numpy as np
#import tensorflow.keras as tfk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from utils.metrics import OME, MSE, get_CV_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from ViscNN import create_ViscNN_concat, ViscNN_concat_HP
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data.dataloader import DataLoader
from utils.model_utils import predict_all_cv, load_models
from utils.train_utils import test_PIHN, polymer_train_test_split, custom_train_test_split, assign_sample_ids, hyperparam_opt, hyperparam_opt_ray
from train_torch import MVDataset, run_training
import keras_tuner as kt
from gpflow_tools.gpr_models import train_GPR, create_GPR
from data_tools.dim_red import fp_PCA
from data_tools.data_viz import val_epochs
import datetime
import keras.backend as K
import random
import yaml
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
import os
import time
import joblib
from utils.Gpuwait import GpuWait

parser = argparse.ArgumentParser(description='Get training vars')
parser.add_argument('--config', default='./config.yaml', help = 'get config file')

def main(file = None):
    global args
    args = parser.parse_args()
    if not file:
        file = args.config
    with open(file) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #USE GPU

    with GpuWait(10, 3600*10, 0.9) as wait:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


    # get data from excel
    data = pd.read_excel(args.file)
    data.columns = [str(c) for c in data.columns]
    if args.aug == True and 'level_0' in data.columns:
        data = data.drop(columns = 'level_0')
    
    ids = {'shear': [], 'Mw': []}
    data, ids['shear'], ids['Mw'] = assign_sample_ids(data.copy())

    
    OG_fp = []
    for c in data.columns:
        if isinstance(c, str):
            if 'fp' in c:
                OG_fp.append(c)
    len(OG_fp)


    #Data Processing #############################################
    if args.do_pca:
        data, fp_cols, pca = fp_PCA(data, 17, cols = OG_fp)
        cols = fp_cols + ['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity', 'PDI']
    else:
        fp_cols = OG_fp
        cols = fp_cols + ['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity', 'PDI', 'Polymer', 'SHEAR', 'ZERO_SHEAR', 'Sample_Type', 'SMILES', 'SAMPLE_ID']
    
    for c in ['Mw', 'Melt_Viscosity']:
        data[c] = np.log10(data[c])

    data['ZERO_SHEAR'] = 1
    data['SHEAR'] = 0
    data['log_Shear_Rate'] = 0
    for i in data.index:
        if data.loc[i, 'Shear_Rate'] != 0:
            data.loc[i,'log_Shear_Rate'] = np.log10(data.loc[i, 'Shear_Rate'])
            data.loc[i, 'SHEAR'] = 1
            data.loc[i, 'ZERO_SHEAR'] = 0
        if not data.loc[i,'PDI'] > 0:
            data.loc[i,'PDI'] = 2.06
        if data.loc[i,'PDI'] > 100:
            data.loc[i,'PDI'] = 2.06
            #data = data.drop([i])

    #################################################################

    #get filtered data
    filtered_data = data.loc[:, cols].dropna(subset = ['Mw', 'Shear_Rate'])

    #Create Test-Train Split#########################################
    if args.full_data:
        train_df = filtered_data.sample(frac = 1)
        test_df = filtered_data.sample(frac = 1) #dummy df for compatibility
    else:
        if args.load_data:
            train_df = pd.read_pickle(f'MODELS/{args.data_date}_{args.data_type}/train_data.pkl')
            if args.data_type == 'full':
                test_df = filtered_data.sample(frac = 0.05) #dummy df for compatibility
            else:
                test_df = pd.read_pickle(f'MODELS/{args.data_date}_{args.data_type}/test_data.pkl')
        else: 
            if args.data_split_type == 'custom':
                total_samps = len(ids['Mw']) + len(ids['shear'])
                leave_out_id = [0]
                # Get list of already completed test ids
                completed_ids = [float(f.split('[', 1)[1].split(']')[0]) for f in os.listdir('./MODELS') if 'custom' in f]
                print(f'comp ids {completed_ids}')
                while (data.loc[data['SAMPLE_ID'] == leave_out_id[0]].shape[0] < 5 or data.loc[data['SAMPLE_ID'] == leave_out_id[0]].shape[0] > 10) and leave_out_id[0] not in completed_ids:
                    leave_out_id = random.sample(ids[args.leave_out],1)
                    print(data.loc[data['SAMPLE_ID'] == leave_out_id[0]])
                train_df, test_df = custom_train_test_split(filtered_data, test_id = leave_out_id, id_col= 'SAMPLE_ID')
                args.data_type = f'{args.data_type}_{args.leave_out}_{leave_out_id}'
            
            elif args.data_split_type == 'polymer':
                train_df, test_df = polymer_train_test_split(filtered_data, test_size = args.test_size, hold_out= args.hold_out)
                print('SHEAR_TEST_post split', np.max(test_df['Shear_Rate']), np.min(test_df['Shear_Rate']))
            elif args.data_split_type == 'random':
                train_df, test_df = train_test_split(filtered_data, test_size = args.test_size)
            
            train_df = train_df.loc[:, (train_df != 0).any(axis=0)]
            new_fp = []
            for c in train_df.columns:
                if isinstance(c, str):
                    if 'fp' in c:
                        new_fp.append(c)

            if len(OG_fp) != len(new_fp):
                test_df = test_df.drop(columns = list(set(OG_fp) - set(new_fp)))
    ####################################################################################

    
    print(len(train_df))
    print(len(test_df))
    #Scale Variables ################################################################# 
    #Only fit scaling on train set and not the test set
    logMw = np.array(train_df['Mw'], dtype = float).reshape((-1,1))
    shear = np.array(train_df['Shear_Rate'], dtype = float).reshape((-1,1))
    print('SHEAR', np.max(shear), np.min(shear))
    Temp = np.array(train_df['Temperature']).reshape((-1,1))
    PDI = np.array(train_df['PDI']).reshape((-1,1))

    scaler = MinMaxScaler(copy = False, feature_range = (0,1))
    XX = np.array(scaler.fit(train_df.filter(fp_cols)).transform(train_df.filter(fp_cols)))
    yy = np.array(train_df.loc[:,'Melt_Viscosity']).reshape((-1,1))

    y_scaler = MinMaxScaler(feature_range = (-1,1)).fit(yy)
    yy = y_scaler.transform(yy);
    T_scaler = MinMaxScaler(feature_range = (0,1)).fit(Temp)
    T = T_scaler.transform(Temp);
    M_scaler = MinMaxScaler(feature_range = (0,1)).fit(logMw)
    M_torch_scaler = MinMaxScaler(feature_range = (0,1)).fit(np.power(10, logMw))
    M = M_scaler.transform(logMw);
    #S_trans = PowerTransformer(standardize = False).fit(shear)
    S_scaler = MinMaxScaler(feature_range = (0,1)).fit(np.log10(shear + 0.00001))
    S = S_scaler.transform(np.log10(shear + 0.00001))
    S_torch_scaler = MinMaxScaler(feature_range = (0,1)).fit(shear)
    P_scaler = MinMaxScaler(feature_range = (0,1)).fit(PDI)
    P = P_scaler.transform(PDI)
    #shear = S_scaler.transform((shear))
    #gpr_Mcr, mcr_cv_error = Mcr_gpr_train(OG_fp, None, M_scaler, scaler, transform = False)
    
    y_test = y_scaler.transform(np.array(test_df.loc[:,'Melt_Viscosity']).reshape((-1,1)))
    X_test = np.array(scaler.transform(test_df.filter(fp_cols)))
    M_test = M_scaler.transform(np.array(test_df['Mw']).reshape((-1,1)))
    S_test = S_scaler.transform(np.log10(0.00001 + np.array(test_df['Shear_Rate']).reshape((-1,1))))
    print('SHEAR_TEST', np.max(test_df['Shear_Rate']), np.min(test_df['Shear_Rate']))
    print('SHEAR_TEST_trans', np.max(S_test), np.min(S_test))
    T_test = np.array(test_df['Temperature']).reshape((-1,1))
    T_test = T_scaler.transform(T_test)
    P_test = P_scaler.transform(np.array(test_df['PDI']).reshape((-1,1)))
    #################################################################################

    #TRAINING #######################################################################
    path = f'../MODELS/{datetime.date.today()}_{args.data_type}'
    os.makedirs(path, exist_ok = True)
    os.makedirs(path + '/scalers/', exist_ok = True)
    
    #Save the scalers in the directory
    
    for sc, n in zip([y_scaler, M_scaler, S_scaler, T_scaler, P_scaler, scaler, S_torch_scaler], ['y_scaler', 'M_scaler', 'S_scaler', 'T_scaler', 'P_scaler', 'scaler', 'S_torch_scaler']):
        sc_name = n + '.save'
        joblib.dump(sc, path + '/scalers/' + sc_name)

    #test_PIHN(XX, yy, M=M, S=S, T=T, P = P, scalers = {"M": M_scaler, "S": S_scaler, "y": y_scaler, "s_torch": S_torch_scaler, "T":T_scaler})
    
    print(XX,yy, M,S,T,P)
    if not args.full_data:
        train_df.to_pickle(f'{path}/train_data.pkl')
        test_df.to_pickle(f'{path}/test_data.pkl')
        models, history, gpr_models, gp_cv, NN_cv, PNN, PNN_cv, PNN_hist  = crossval_compare([create_ViscNN_concat], XX, yy, M=M, S=S, T=T, P = P, verbose = 1, gpr_model = create_GPR, epochs = 600, train_type=args.data_type, scalers = {"M": M_scaler, "S": S_scaler, "y": y_scaler, "s_torch": S_torch_scaler, "T":T_scaler, "m_torch":M_torch_scaler})
    else:
        train_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/train_data.pkl')
        models_f, history_f, gpr_models_f, gp_cv_f, NN_cv, PNN, PNN_cv, PNN_hist = crossval_compare([create_ViscNN_concat], XX, yy, M=M, S=S, T=T, P =P, verbose = 1, gpr_model = create_GPR, epochs = 600, train_type=args.data_type, scalers = {"M": M_scaler, "S": S_scaler, "y": y_scaler, "s_torch": S_torch_scaler, "T":T_scaler, "m_torch":M_torch_scaler})
    #################################################################################

    # PLOT PARITY ###################################################################
    if not args.full_data:
        yy = y_scaler.inverse_transform(yy)
        y_test = y_scaler.inverse_transform(y_test)
        
        test_pred, test_var, _ = predict_all_cv(models[0],[X_test, M_test, S_test, T_test, P_test])
        train_pred, train_var, _ = predict_all_cv(models[0],[XX, M, S, T, P])
        test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
        train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
        test_var = y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1) - 1) - y_scaler.data_min_
        train_var = y_scaler.inverse_transform(np.array(train_var).reshape(-1, 1) - 1) - y_scaler.data_min_

        train_df["ANN_Pred"] = train_pred
        test_df["ANN_Pred"] = test_pred
        train_df["ANN_Pred_var"] = train_var
        test_df["ANN_Pred_var"] = test_var

        plt.figure(figsize = (5.5,5.5))
        plt.errorbar(yy, list(train_pred.reshape(-1,)), yerr = list(np.array(train_var).reshape(-1,)), c = 'orange', fmt = 'o', label = f'Train: {M.shape[0]} datapoints, ' 
        + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(yy, train_pred)) + "OME = {:1.4f}".format(get_CV_error(NN_cv, scaler= y_scaler)))

        plt.errorbar(y_test , list(test_pred.reshape(-1,)), yerr= list(np.array(test_var).reshape(-1,)), fmt =  'o', label = f'Test: {M_test.shape[0]} datapoints, ' 
        + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(y_test, test_pred)) + "OME = {:1.4f}".format(OME(y_test, test_pred)))

        plt.plot(np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),'k-', zorder = 10)
        plt.ylabel(r'$Log$ Viscosity (Poise) ML Predicted')
        plt.xlabel(r'$Log$ Viscosity (Poise) Experimental Truth')
        plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
        plt.title('ANN Parity Plot')
        plt.xlim(-2, 12.5)
        plt.ylim(-2, 12.5)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(f'MODELS/{datetime.date.today()}_{args.data_type}/ANN_parity_plot.png')
        val_epochs(history[0], name = 'ANN', scaler = y_scaler, save = True, d_type= args.data_type)


        #GPR Parity
        try:
            gpr_model = gpr_models[np.argmin(gp_cv)]
            X_ = np.concatenate((X_test, M_test, S_test, T_test, P_test), axis = 1)
            X_train = np.concatenate((XX, M,S, T, P), axis = 1)
            test_pred, var = gpr_model.predict_f(tf.convert_to_tensor(X_, dtype=tf.float64))
            train_pred, var_train = gpr_model.predict_f(X_train)
            error =  [i[0] for i in np.array(var).tolist()]
            test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
            train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
            var = y_scaler.inverse_transform(np.array(var).reshape(-1, 1) - 1) - y_scaler.data_min_
            var_train = y_scaler.inverse_transform(np.array(var_train).reshape(-1, 1) - 1) - y_scaler.data_min_
            high_var_test = np.where(np.array(var) > 10)
            test_pred = np.delete(np.array(test_pred), high_var_test)
            y_test_new = np.delete(np.array(y_test), high_var_test)
            var = np.delete(np.array(var), high_var_test)
            # plt.plot(xx[:,0], samples[:, :, 0].numpy().T, "C0", linewidth=0.5)

            test_pred = test_pred.reshape(-1,1)
            train_pred = train_pred.reshape(-1,1)
            train_df["GPR_Pred"] = train_pred
            test_df["GPR_Pred"] = test_pred
            train_df["GPR_Pred_var"] = var_train
            test_df["GPR_Pred_var"] = var

            plt.figure(figsize = (5.5,5.5))
            plt.errorbar(yy, list(train_pred.reshape(-1,)), yerr = list(np.array(var_train).reshape(-1,)), c = 'orange', fmt = 'o', label = f'Train: {M.shape[0]} datapoints, ' 
            + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score( yy, train_pred)) + "OME = {:1.4f}".format(get_CV_error(gp_cv, scaler= y_scaler)))

            plt.errorbar(y_test , list(test_pred.reshape(-1,)), yerr= list(np.array(var).reshape(-1,)), fmt =  'o', label = f'Test: {M_test.shape[0]} datapoints, ' 
            + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(y_test, test_pred)) + "OME = {:1.4f}".format(OME(y_test, test_pred)))


            plt.plot(np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),'k-', zorder = 10)
            plt.ylabel(r'$Log$ Viscosity (Poise) ML Predicted')
            plt.xlabel(r'$Log$ Viscosity (Poise) Experimental Truth')
            plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
            plt.title('GPR Parity Plot')
            plt.xlim(-2, 12.5)
            plt.ylim(-2, 12.5)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(f'MODELS/{datetime.date.today()}_{args.data_type}/GPR_parity_plot.png')
        except:
            print(f'GPR parity could not be made for {datetime.date.today()}_{args.data_type}')
        
        # test_pred, test_var, _ = predict_all_cv(PNN,[X_test, y_scaler.transform(M_scaler.inverse_transform(M_test)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S_test)) - 0.00001), T_test, P_test], is_torch = True)
        # train_pred, train_var, _ = predict_all_cv(PNN,[XX, y_scaler.transform(M_scaler.inverse_transform(M)), S_torch_scaler.transform(np.power(10, S_scaler.inverse_transform(S)) - 0.00001), T, P], is_torch = True)
       
        #With Scaling
        try:
            test_pred, test_var, _ = predict_all_cv(PNN,[X_test, y_scaler.transform(M_scaler.inverse_transform(M_test)), y_scaler.transform(S_scaler.inverse_transform(S_test)), T_test, P_test], is_torch = True)
            train_pred, train_var, _ = predict_all_cv(PNN,[XX, y_scaler.transform(M_scaler.inverse_transform(M)), y_scaler.transform(S_scaler.inverse_transform(S)), T, P], is_torch = True)
            test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
            train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
            test_var = y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1) - 1) - y_scaler.data_min_
            train_var = y_scaler.inverse_transform(np.array(train_var).reshape(-1, 1) -1) - y_scaler.data_min_
        except:
            print('Error in PIHN predictions.')
            
        #Without log Scaling
        # test_pred, test_var, _ = predict_all_cv(PNN,[X_test, M_torch_scaler.transform(np.power(10, (M_scaler.inverse_transform(M_test)))), y_scaler.transform(S_scaler.inverse_transform(S_test)), T_test, P_test], is_torch = True)
        # train_pred, train_var, _ = predict_all_cv(PNN,[XX, M_torch_scaler.transform(np.power(10, (M_scaler.inverse_transform(M)))), y_scaler.transform(S_scaler.inverse_transform(S)), T, P], is_torch = True)
        # test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
        # train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
        # test_var = y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)) - y_scaler.data_min_
        # train_var = y_scaler.inverse_transform(np.array(train_var).reshape(-1, 1)) - y_scaler.data_min_

        train_df["PENN_Pred"] = train_pred
        test_df["PENN_Pred"] = test_pred
        train_df["PENN_Pred_var"] = train_var
        test_df["PENN_Pred_var"] = test_var

        plt.figure(figsize = (5.5,5.5))
        plt.errorbar(yy, list(train_pred.reshape(-1,)), yerr = list(np.array(train_var).reshape(-1,)), c = 'orange', fmt = 'o', label = f'Train: {M.shape[0]} datapoints, ' 
        + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(yy, train_pred)) + "OME = {:1.4f}".format(get_CV_error(PNN_cv, scaler= y_scaler)))

        plt.errorbar(y_test , list(test_pred.reshape(-1,)), yerr= list(np.array(test_var).reshape(-1,)), fmt =  'o', label = f'Test: {M_test.shape[0]} datapoints, ' 
        + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(y_test, test_pred)) + "OME = {:1.4f}".format(OME(y_test, test_pred)))

        plt.plot(np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),np.linspace((min(yy)[0]), (max(yy)[0]), num = 2),'k-', zorder = 10)
        plt.ylabel(r'$Log$ Viscosity (Poise) ML Predicted')
        plt.xlabel(r'$Log$ Viscosity (Poise) Experimental Truth')
        plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
        plt.title('PNN Parity Plot')
        plt.xlim(-2, 12.5)
        plt.ylim(-2, 12.5)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(f'MODELS/{datetime.date.today()}_{args.data_type}/PNN_parity_plot.png')
        val_epochs(PNN_hist, name = 'PNN', scaler = y_scaler, save = True, d_type= args.data_type)
    #################################################################################

    train_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/train_evals.pkl')
    test_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/test_evals.pkl')
        
def crossval_compare(NN_models, XX, yy, M, S, T, P, data = None, custom = False, verbose = 1, random_state = None, epochs = 500, gpr_model = None, save = True, train_type = 'split', scalers = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    start_total = time.time()

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
            np.save(f'MODELS/{datetime.date.today()}_{train_type}/PNN/OME_CV', np.array(PNN_CV))
            print(f'Saved PNN_{i}')
        for i in range(len(gpr)):
            gpr[i].predict_f_compiled = tf.function(gpr[i].predict_f, input_signature=[tf.TensorSpec(shape=[None, X_train_.shape[1]], dtype=tf.float64)])
            tf.saved_model.save(gpr[i], f'MODELS/{datetime.date.today()}_{train_type}/GPR/model_{i}')
            print(f'Saved GPR_{i}')
        np.save(f'MODELS/{datetime.date.today()}_{train_type}/GPR/OME_CV', np.array(gp_cv))
        print('saved models')
    return NN, hist, gpr, gp_cv, NN_cv, PNN, PNN_CV, PNN_hist

if __name__ == '__main__':
    main()