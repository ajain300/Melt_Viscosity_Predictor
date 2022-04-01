import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from validation.metrics import OME, MSE, get_CV_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from MODELS.ViscNN import load_models, create_ViscNN_concat, create_ViscNN_phys,predict_all_cv, ViscNN_concat_HP,  create_ViscNN_phys_HP
from validation.tests import custom_train_test_split, get_Mw_samples, crossval_NN, Mw_test, evaluate_model, crossval_compare, get_shear_samples, shear_test, small_shear_test, assign_sample_ids
import keras_tuner as kt
from gpflow_tools.gpr_models import train_GPR, create_GPR
from data_tools.dim_red import fp_PCA
from data_tools.data_viz import val_epochs, calc_slopes_Mw, compare_cv
import datetime
import keras.backend as K
import keras
import random
import yaml
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
import os

parser = argparse.ArgumentParser(description='Get training vars')
parser.add_argument('--config', default='./config.yaml', help = 'get config file')

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #USE GPU
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
    
    data, shear_ids, Mw_ids = assign_sample_ids(data.copy())

    
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
            data.loc[i,'PDI'] = 2
        if data.loc[i,'PDI'] > 100:
            data.loc[i,'PDI'] = 2
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
            if args.custom_data_split:
                total_samps = len(Mw_ids) + len(shear_ids)
                train_df, test_df = custom_train_test_split(filtered_data, test_id= random.sample(Mw_ids,total_samps//20) + random.sample(shear_ids,total_samps//20), id_col= 'SAMPLE_ID')
            else:
                train_df, test_df = train_test_split(filtered_data, test_size= args.test_size)
            
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
    logMw = np.array(train_df['Mw']).reshape((-1,1))
    shear = np.array(train_df['Shear_Rate']).reshape((-1,1))
    Temp = np.array(train_df['Temperature']).reshape((-1,1))
    Temp = 1/(Temp+273.15)
    PDI = np.array(train_df['PDI']).reshape((-1,1))

    scaler = MinMaxScaler(copy = False)
    XX = np.array(scaler.fit(train_df.filter(fp_cols)).transform(train_df.filter(fp_cols)))
    yy = np.array(train_df.loc[:,'Melt_Viscosity']).reshape((-1,1))

    y_scaler = MinMaxScaler().fit(yy)
    yy = y_scaler.transform(yy);
    T_scaler = MinMaxScaler().fit(Temp)
    T = T_scaler.transform(Temp);
    M_scaler = MinMaxScaler().fit(logMw)
    M = M_scaler.transform(logMw);
    S_trans = PowerTransformer(standardize = False).fit(shear)
    S_scaler = MinMaxScaler().fit(S_trans.transform(shear))
    S = S_scaler.transform(S_trans.transform(shear))
    P_scaler = MinMaxScaler().fit(PDI)
    P = P_scaler.transform(PDI)
    #shear = S_scaler.transform((shear))
    #gpr_Mcr, mcr_cv_error = Mcr_gpr_train(OG_fp, None, M_scaler, scaler, transform = False)
    
    y_test = y_scaler.transform(np.array(test_df.loc[:,'Melt_Viscosity']).reshape((-1,1)))
    X_test = np.array(scaler.transform(test_df.filter(fp_cols)))
    M_test = M_scaler.transform(np.array(test_df['Mw']).reshape((-1,1)))
    S_test = S_scaler.transform(S_trans.transform(np.array(test_df['Shear_Rate']).reshape((-1,1))))
    T_test = np.array(test_df['Temperature']).reshape((-1,1))
    T_test = T_scaler.transform(1/(T_test+273.15))
    P_test = P_scaler.transform(np.array(test_df['PDI']).reshape((-1,1)))
    #################################################################################

    #TRAINING #######################################################################
    os.makedirs(f'MODELS/{datetime.date.today()}_{args.data_type}', exist_ok = True)
    if not args.full_data:
        train_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/train_data.pkl')
        test_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/test_data.pkl')
        models, history, gpr_models, gp_cv, NN_cv  = crossval_compare([create_ViscNN_concat], XX, yy, M=M, S=S, T=T, P = P, verbose = 1, gpr_model = create_GPR, epochs = 600, train_type=args.data_type)
    else:
        train_df.to_pickle(f'MODELS/{datetime.date.today()}_{args.data_type}/train_data.pkl')
        models_f, history_f, gpr_models_f, gp_cv_f, NN_cv = crossval_compare([create_ViscNN_concat], XX, yy, M=M, S=S, T=T, P =P, verbose = 1, gpr_model = create_GPR, epochs = 600, train_type=args.data_type)
    #################################################################################

    # PLOT PARITY ###################################################################
    if not args.full_data:
        yy = y_scaler.inverse_transform(yy)
        y_test = y_scaler.inverse_transform(y_test)
        
        test_pred, test_var, _ = predict_all_cv(models[0],[X_test, M_test, S_test, T_test, P_test])
        train_pred, train_var, _ = predict_all_cv(models[0],[XX, M, S, T, P])
        test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
        train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
        test_var = y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)) - y_scaler.data_min_
        train_var = y_scaler.inverse_transform(np.array(train_var).reshape(-1, 1)) - y_scaler.data_min_


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
            gpr_model = gpr_models[5] 
            X_ = np.concatenate((X_test, M_test, S_test, T_test, P_test), axis = 1)
            X_train = np.concatenate((XX, M, S, T, P), axis = 1)
            test_pred, var = gpr_model.predict_f_compiled(tf.convert_to_tensor(X_, dtype=tf.float64))
            train_pred, var_train = gpr_model.predict_f_compiled(X_train)
            error =  [i[0] for i in np.array(var).tolist()]
            test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
            train_pred = y_scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
            var = y_scaler.inverse_transform(np.array(var).reshape(-1, 1)) - y_scaler.data_min_

            high_var_test = np.where(np.array(var) > 10)
            test_pred = np.delete(np.array(test_pred), high_var_test)
            y_test_new = np.delete(np.array(y_test), high_var_test)
            var = np.delete(np.array(var), high_var_test)
            # plt.plot(xx[:,0], samples[:, :, 0].numpy().T, "C0", linewidth=0.5)


            test_pred = test_pred.reshape(-1,1)
            train_pred = train_pred.reshape(-1,1)

            plt.figure(figsize = (5.5,5.5))
            plt.errorbar(yy, list(train_pred.reshape(-1,)), yerr = list(np.array(train_var).reshape(-1,)), c = 'orange', fmt = 'o', label = f'Train: {M.shape[0]} datapoints, ' 
            + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score( yy, train_pred)) + "OME = {:1.4f}".format(get_CV_error(gp_cv, scaler= y_scaler)))

            plt.errorbar(y_test , list(test_pred.reshape(-1,)), yerr= list(np.array(test_var).reshape(-1,)), fmt =  'o', label = f'Test: {M_test.shape[0]} datapoints, ' 
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
    #################################################################################

if __name__ == '__main__':
    main()