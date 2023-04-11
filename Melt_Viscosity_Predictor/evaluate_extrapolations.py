import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from validation.metrics import OME, MSE, get_CV_error
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from .ViscNN import load_models, create_ViscNN_concat, predict_all_cv, ViscNN_concat_HP
from validation.tests import custom_train_test_split, get_Mw_samples, crossval_NN, Mw_test, evaluate_model, crossval_compare, get_shear_samples, get_temp_samples, temp_test, shear_test, small_shear_test, assign_sample_ids
import keras_tuner as kt
from gpflow_tools.gpr_models import train_GPR, create_GPR
from data_tools.dim_red import fp_PCA
from data_tools.data_viz import val_epochs, calc_slopes_Mw, compare_cv
from data_tools.curve_fitting import *
from data_tools.SeabornFig2Grid import SeabornFig2Grid as sfg
import datetime
import keras.backend as K
import pickle
from math import floor
import joblib
import tensorflow_text as text

from textwrap import wrap
pd.options.mode.chained_assignment = None  # default='warn'
import os


class ModelLoadingError(Exception):
    "Raised when a certain model was unable to load"
    pass

tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device= '/job:localhost',
    experimental_skip_checkpoint=False,
    experimental_variable_policy=None
)

model_loading_errors = []
predicted_data = pd.DataFrame()
models_run = 0
samples_run = 0
fitting_error = pd.DataFrame(np.zeros([3,3]), index = ['ANN', 'GPR', 'PIMI'], columns = ['Mw', 'Shear', 'Temperature'])

Mw_const = {'GPR': pd.DataFrame(columns = ['Sample', 'a1', 'a2', 'k1', 'k2', 'Mcr', 'r2']),
'ANN': pd.DataFrame(columns = ['Sample', 'a1', 'a2', 'k1', 'k2', 'Mcr', 'r2']), 
'PIM':pd.DataFrame(columns = ['Sample', 'a1', 'a2', 'k1', 'k2', 'Mcr', 'r2'])}

shear_const = {'GPR': pd.DataFrame(columns = ['Sample', 'z_shear', 'n', 'S_cr', 'r2']),
'ANN': pd.DataFrame(columns = ['Sample', 'z_shear', 'n', 'S_cr', 'r2']), 
'PIM': pd.DataFrame(columns = ['Sample', 'z_shear', 'n', 'S_cr', 'r2'])}

WLF_const = {'GPR': pd.DataFrame(columns = ['Sample','C1', 'C2', 'Tr', 'r2']),
'ANN': pd.DataFrame(columns = ['Sample','C1', 'C2', 'Tr', 'r2']), 
'PIM': pd.DataFrame(columns = ['Sample','C1', 'C2', 'Tr', 'r2'])}


def main(date, data_type, save_path, plot = True):
    global samples_run
    global models_run
    global model_loading_errors
    global predicted_data
    global fitting_error
    with open(f'MODELS/{date}_{data_type}/test_evals.pkl', 'rb') as f:
        data= pickle.load(f)
    with open(f'MODELS/{date}_{data_type}/test_data.pkl', 'rb') as f:
        test_data= pickle.load(f)
    OG_data = data.copy()
    #data = pd.read_excel('./Data/full_data_03_08_aug_MOD.xlsx', na_values = ['nan','','NaN'])
    #data = pd.read_excel('./Data/validation_data_2022-09-06.xlsx', na_values = ['nan','','NaN'])
    data.columns = [str(c) for c in data.columns]

    OG_fp = []
    for c in data.columns:
        if isinstance(c, str):
            if 'fp' in c:
                OG_fp.append(c)
    print('EQUAL FP',  test_data[OG_fp].equals(data[OG_fp]))

    fp_cols = OG_fp
    cols = fp_cols + ['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity']

    #data.loc[data['Mw'] < 10, 'Mw'] = np.power(10, data.loc[data['Mw'] < 10, 'Mw'])
    #for c in ['Mw', 'Melt_Viscosity']:
    #    data[c] = np.log10(data[c])

    data['PDI'].fillna(2)
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
    
    if max(data['Mw']) > 1000:
        data['Mw'] = np.log10(data['Mw'])
    
    if max(data['Melt_Viscosity']) > 1000:
        data['Melt_Viscosity'] = np.log10(data['Melt_Viscosity'])


    #Reload models


    models_f, history_f, gpr_models_f, gp_cv, NN_cv, HypNet_f, HypNet_cv = load_models(date = date, data_type = data_type, NN_models = [create_ViscNN_concat])
    models_f = models_f[0]
    model_reloaded = True

    if len(models_f) == 0 or len(gpr_models_f) == 0 or len(HypNet_f) == 0:
        model_loading_errors.append(f'{date}_{data_type}')
        raise ModelLoadingError

    #Load scalers
    y_scaler, M_scaler, S_scaler, T_scaler, P_scaler, scaler, S_torch_scaler = joblib.load(f'MODELS/{date}_{data_type}/scalers/y_scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/M_scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/S_scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/T_scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/P_scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/scaler.save'), joblib.load(f'MODELS/{date}_{data_type}/scalers/S_torch_scaler.save')
    print('data_in', data[['Polymer', 'Mw', 'Shear_Rate', 'Temperature']])
        
    #Create samples
    #data['Melt_Viscosity'] = y_scaler.inverse_transform(np.array(data['Melt_Viscosity']).reshape(-1,1))
    #print('FOUND DATA',data)
    samples_run += len(data)
    Mw_samps, mw_ids = get_Mw_samples(data.copy().reset_index(drop = True), full = True)
    shear_samps, shear_ids = get_shear_samples(data.copy().reset_index(drop = True), full = True)
    temp_samps, temp_ids = get_temp_samples(data.copy().reset_index(drop = True), full = True)

    # print('MV max', max(data['Melt_Viscosity']))
    # print('MV min', min(data['Melt_Viscosity']))
    # print('shear max', max(data['Shear_Rate']))
    # print('shear min', min(data['Shear_Rate']))

    # print('Mw_ids', Mw_samps[['SAMPLE_ID', 'Polymer', 'Mw', 'Temperature']].head(50))
    # print('Shear samps', shear_samps.head(50))

    
    t_ranges = [20,60]
    
    DO_TOTAL = False
    DO_MW = True & (len(Mw_samps)!= 0)
    if DO_MW == False:
        print('No zero shear', data[['Polymer','Mw', 'Shear_Rate', 'Temperature']])
    DO_SHEAR =True & (len(shear_samps) != 0)
    DO_TEMP = True & (len(temp_samps) != 0)

    if DO_TOTAL: #for gpr fix an d
        logMw = np.array(OG_data['Mw']).reshape((-1,1))
        shear = np.array(OG_data['Shear_Rate']).reshape((-1,1))
        Temp = np.array(OG_data['Temperature']).reshape((-1,1))
        Temp = Temp
        PDI = np.array(OG_data['PDI']).reshape((-1,1))

        XX = scaler.transform(OG_data.filter(fp_cols))
        yy = np.array(OG_data.loc[:,'Melt_Viscosity']).reshape((-1,1))
        yy = y_scaler.transform(yy);
        T = T_scaler.transform(Temp);
        M = M_scaler.transform(logMw);
        S = S_scaler.transform(np.log10(shear + 0.00001))
        P = P_scaler.transform(PDI)

        OG_data.to_pickle(f'MODELS/{date}_{data_type}/test_evals_old.pkl')
        test_pred, var = gpr_models_f[np.argmin(gp_cv)-1].predict_f_compiled(np.concatenate((XX, M, S, T, P), axis = 1))
        print('GPR PRED', test_pred)
        test_pred = y_scaler.inverse_transform(np.array(test_pred))
        print('GPR VAR', var)
        var = y_scaler.inverse_transform(np.array(var).reshape(-1, 1) - 1) - y_scaler.data_min_

        OG_data['GPR_Pred'] = test_pred
        OG_data['GPR_Pred_var'] = var

        OG_data.to_pickle(f'MODELS/{date}_{data_type}/test_evals.pkl')
        exit()


    #Directories
    path = f'{save_path}/{date}_{data_type}/'
    # os.makedirs(path + 'Mw/GPR', exist_ok = True)
    # os.makedirs(path + 'Mw/ANN', exist_ok = True)
    # os.makedirs(path + 'Mw/HypNet', exist_ok = True)
    # os.makedirs(path + 'shear/GPR', exist_ok = True)
    # os.makedirs(path + 'shear/ANN', exist_ok = True)
    # os.makedirs(path + 'shear/HypNet', exist_ok = True)
    # os.makedirs(path + 'temp', exist_ok = True)
    # os.makedirs(path + 'temp/test', exist_ok = True)
    # os.makedirs(path + 'Mw/saved_extraps/', exist_ok = True)
    


    matplotlib.rc('xtick', labelsize= 30)
    matplotlib.rc('ytick', labelsize= 30)
    matplotlib.rc('axes', labelsize= 35) 

    

    colors = {'GPR':plt.get_cmap('Accent')(0.3), 'ANN': plt.get_cmap('Accent')(0.2), 'PIHN': plt.get_cmap('Accent')(0.1), 'Data': plt.get_cmap('Accent')(0.5)}

    print('Starting Mw extrapolations...')
    k1 = 0


    if DO_MW:
        for mid in mw_ids:#list(range(1022,1034)):
            try:
                extraps = np.load(path + f'Mw/saved_extraps/{int(mid)}.npz')
                extrap_exists = True
                print(f'Extrapolation found for {int(mid)}.')
            except:
                extrap_exists = False
                print(f'No extrapolation found for {int(mid)}.')

            extrap_tests = Mw_test(Mw_samps, mid)
            XX_ex, OH, M_ex_og,S_ex_og,T_ex_og, P_ex_og = extrap_tests['data_in']
            log_visc_ex = extrap_tests['exp'][1][~np.isnan(np.power(10,extrap_tests['exp'][1]))]
            Mw_ex = extrap_tests['exp'][0][~np.isnan((extrap_tests['exp'][1]))]

            #Mw_ex = np.log10(Mw_ex)
            if Mw_ex.shape[0] < 5 and plot:
                continue
            os.makedirs(path + 'Mw/HypNet', exist_ok = True)
            os.makedirs(path + 'Mw/saved_extraps/', exist_ok = True)
            print('OG datapoints num', Mw_ex.shape)
            print('original viscosities', log_visc_ex)
            P_exp = extrap_tests['exp'][2][~np.isnan((extrap_tests['exp'][1]))]
            P_exp = P_scaler.transform(np.array(P_exp).reshape(-1,1))
            
            print('XX_ex',XX_ex, 'Mol weight extrap', M_ex_og, 'shear rate extrap_Mw', S_ex_og, 'temp extrap', T_ex_og)
            print('Mw_ex', Mw_ex)
            XX_ex = scaler.transform((XX_ex))
            M_ex = M_scaler.transform(M_ex_og)
            S_ex = S_scaler.transform(np.log10(S_ex_og+0.00001))
            S_ex_torch_2 = y_scaler.transform(np.log10(S_ex_og+0.00001))
            T_ex = T_scaler.transform(T_ex_og)
            P_ex = P_scaler.transform(P_ex_og)
            y_ax = (-2, 11)
            temps = np.linspace(-50, 50, 20)
            #print('M_ex (scaled)', M_ex)
            print('XX_ex_scaled',XX_ex, 'Mol weight extrap_scaled', M_ex, 'shear rate extrap_scaled', S_ex, 'temp extrap_scaled', T_ex)
            #start building figure
            fig = plt.figure(figsize = (7,10))
            fig.suptitle(f"HyperNet Prediction - {extrap_tests['sample']}")
            
            ax1 = plt.subplot(1,1,1)
            ax1.set_box_aspect(1)
            ax = [ax1]
            tick = 0

            #GPR Extrapolation
            l = Mw_ex.shape[0]
            if extrap_exists:
                test_pred = extraps['gpr_pred']
                var = extraps['gpr_var']
            else:
                print('input to gpr',np.concatenate((XX_ex, M_ex, S_ex, T_ex, P_ex), axis = 1)[0])
                test_pred, var = gpr_models_f[np.argmin(gp_cv)].predict_f_compiled(np.concatenate((XX_ex, M_ex, S_ex, T_ex, P_ex), axis = 1))
                gpr_tried = 0
                while gpr_tried < 10:
                    if tf.math.is_nan(test_pred).numpy().any():
                        test_pred, var = gpr_models_f[gpr_tried].predict_f_compiled(np.concatenate((XX_ex, M_ex, S_ex, T_ex, P_ex), axis = 1))
                        gpr_tried += 1
                    else:
                        gpr_tried = 10
                print('GPR PRED', test_pred)
                test_pred = y_scaler.inverse_transform(np.array(test_pred))
                print('GPR VAR', var)
                var = y_scaler.inverse_transform(np.array(var).reshape(-1, 1) - 1) - y_scaler.data_min_
                #var = np.array(var)
                gpr_pred = test_pred
                gpr_var = var
            
            test_pred_og = test_pred
            #a1, a2, Mcr, k2 = fit_Mw_3(np.array([a[0] for a in M_ex_og]).astype(float), np.array([a[0] for a in test_pred]).astype(float))
            a1, a2, k1, k2, Mcr, r2 = fit_Mw(np.array([a[0] for a in M_ex_og]).astype(float), np.array([a[0] for a in test_pred]).astype(float))
            Mw_const['GPR'] = pd.concat([Mw_const['GPR'], pd.DataFrame({'Sample': extrap_tests['sample'], 'a1': [a1], 'a2':[a2], 'k1':[k1], 'k2':[k2], 'Mcr':[Mcr], 'r2':[r2]})], ignore_index= True)

            line1 = ax[tick].plot(M_scaler.inverse_transform(M_ex), test_pred.reshape(-1,), '--',label = f'GPR Prediction', color = colors['GPR'])
            plt.fill_between(M_scaler.inverse_transform(M_ex).reshape(-1,), (test_pred- var).reshape(-1,), (test_pred + var).reshape(-1,), alpha = 0.2, color = colors['GPR'])


            #ANN Extrapolation
            l = Mw_ex.shape[0]
            if extrap_exists:
                test_pred = extraps['ann_pred']
                test_var = extraps['ann_var']
            else:
                test_pred, test_var,_ = predict_all_cv(models_f,[XX_ex, M_ex, S_ex, T_ex, P_ex])
                print('ANN test_pred', test_pred)
                test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
                test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1) -1) - y_scaler.data_min_).reshape(-1,)
                ann_pred = test_pred
                ann_var = test_var
            
            #a1, a2, Mcr, k2 = fit_Mw_3(np.array([a[0] for a in M_ex_og]).astype(float), np.array([a for a in test_pred]).astype(float))
            a1, a2, k1, k2, Mcr, r2 = fit_Mw(np.array([a[0] for a in M_ex_og]).astype(float), np.array([a for a in test_pred]).astype(float))
            Mw_const['ANN'] = pd.concat([Mw_const['ANN'], pd.DataFrame({'Sample': extrap_tests['sample'], 'a1': [a1], 'a2':[a2], 'k1':[k1], 'k2':[k2], 'Mcr':[Mcr], 'r2':[r2]})], ignore_index= True)

            test_pred_og = test_pred
            #temperature constants
            line1 = ax[tick].plot(M_scaler.inverse_transform(M_ex), test_pred.reshape(-1,), '--',label = f'ANN Prediction', color = colors['ANN'])
            ax[tick].set_ylabel(r'$\eta$ (Poise)')
            ax[tick].set_xlabel(r'$M_w$ (g/mol)')

            plt.fill_between(M_scaler.inverse_transform(M_ex).reshape(-1,), (test_pred- test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,), alpha = 0.2, color = colors['ANN'])

            ann_aT = []


            #PIHN Exptrapolation

            l = Mw_ex.shape[0]
            # exp_pred, _, _ = predict_all_cv(HypNet_f, [XX_ex[:l], y_scaler.transform(np.array(Mw_ex).reshape(-1,1)), S_ex_torch[:l], T_ex[:l], P_exp], is_torch = True)
            # exp_pred =  y_scaler.inverse_transform(np.array(exp_pred).reshape(-1, 1)).reshape(-1,)
            if extrap_exists:
                test_pred = extraps['pihn_pred']
                test_var = extraps['pihn_var']
                constants = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True, get_constants = True)
            else:
                try:
                    test_pred, test_var, _ = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True)
                    constants = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True, get_constants = True)
                except:
                    continue
                #print(constants)
                test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
                test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)-1) - y_scaler.data_min_).reshape(-1,)
                pihn_pred = test_pred
                pihn_var = test_var
            
            a1, a2, Mcr, kcr = constants["a1"][0], constants["a2"][0], y_scaler.inverse_transform(constants["Mcr"][0].reshape(-1, 1)).item(), y_scaler.inverse_transform(constants["kcr"][0].reshape(-1, 1)).item()
            r2 = r2_score(test_pred, Mw_func_obj(np.array([a[0] for a in M_ex_og]).astype(float),  a1, a2, Mcr, kcr))
            print(r2)
            Mw_const['PIM'] = pd.concat([Mw_const['PIM'], pd.DataFrame({'Sample': extrap_tests['sample'], 'a1': [a1], 'a2':[a2], 'k1':[k1], 'k2':[k2], 'Mcr':[Mcr], 'r2':[r2]})], ignore_index= True)

            test_pred_og = test_pred

            line1 = ax[tick].plot(M_scaler.inverse_transform(M_ex), test_pred.reshape(-1,), '--',label = f'PIHN Prediction', color = colors['PIHN'])
            plt.fill_between(M_scaler.inverse_transform(M_ex).reshape(-1,), (test_pred - test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,) , alpha = 0.2, color = colors['PIHN'])
            
            if plot:
                ax[tick].text(5, 3, r'$\alpha_2=$' + "{:1.4f}".format(a2), fontsize = 13)
                ax[tick].text(5, 1.5, r'$k_2=$' + "{:1.4f}".format(k2), fontsize = 13)
                ax[tick].text(5, -0, r'$Mcr=$' + "{:1.4f}".format(Mcr), fontsize = 13)
                ax[tick].text(2, 6.5, r'$\alpha_1=$' + "{:1.4f}".format(a1), fontsize = 13)
                #ax[tick].text(2, 5, r'$k_1=$' + "{:1.4f}".format(k1), fontsize = 13)

                ax[tick].set_xlabel(r'$M_w$ (g/mol)', labelpad = 0.5)
                ax[tick].set_ylabel(r'$\eta_o$ (Poise)',labelpad = -3)
                ax[tick].set_ylim(-2,10)
                ax[tick].set_xticks(list(np.arange(2,8,2)),[rf'$10^{i}$' for i in list(np.arange(2,8,2))])
                ax[tick].set_yticks(list(np.arange(-2,11,4)), [r'$10^{-2}$'] + [rf'$10^{i}$' for i in list(np.arange(2,10,4))] + [r'$10^{10}$'])

                exp = ax[tick].scatter(Mw_ex, log_visc_ex, c = 'orange', label = 'Experimental Datapoints') #f'Experimental at {T_ex_og[0][0]} C'
                ax[tick].axis(ymin = y_ax[0], ymax = y_ax[1])
                ax[tick].legend(fontsize = 16, loc = 'lower right', bbox_to_anchor=(1, -.75), ncol = 4)
                #ax[tick].plot(M_scaler.inverse_transform(M_ex), Mw_func_obj(M_scaler.inverse_transform(M_ex), a1, a2, Mcr, kcr), color = 'red')
                #plt.savefig(f"../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/Mw/HypNet/{extrap_tests['sample']}_{mid}.svg", bbox_inches = 'tight')
                print(f'saving MW extrap for {data_type}')
                plt.savefig(f"{test_folder}/{date}_{data_type}/Mw/HypNet/{extrap_tests['sample']}_{T_ex_og[0]}_{mid}.png", bbox_inches = 'tight')
                plt.savefig(f"{test_folder}/{date}_{data_type}/Mw/HypNet/{extrap_tests['sample']}_{T_ex_og[0]}_{mid}.svg", bbox_inches = 'tight', dpi = 300)
            if not extrap_exists:
                np.savez(path + f'Mw/saved_extraps/{int(mid)}.npz', gpr_pred = gpr_pred, gpr_var = gpr_var,ann_pred = ann_pred, ann_var = ann_var, pihn_pred = pihn_pred, pihn_var = pihn_var)
            

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/gpr_Mw_const.pickle', 'wb') as handle:
        #     pickle.dump(predicted_constants_gpr, handle)

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/ANN_Mw_const.pickle', 'wb') as handle:
        #     pickle.dump(predicted_constants_ann, handle)

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/HypNet_Mw_const.pickle', 'wb') as handle:
        #     pickle.dump(predicted_constants_HypNet, handle)


    
    print('Saved Mw extrap files')

    print('Starting shear extrapolations...')

    #####Shear
    if DO_SHEAR:
        
        for sid in shear_ids:
            try:
                extrap_tests = shear_test(shear_samps, sid)
                XX_ex, M_ex_og,S_ex_og,T_ex_og, P_ex_og = extrap_tests['data_in']
                
                sample = extrap_tests['sample']
                log_visc_ex = extrap_tests['exp'][1][~np.isnan(np.power(10,extrap_tests['exp'][1]))]
                shear_exp = extrap_tests['exp'][0][~np.isnan((extrap_tests['exp'][1]))]
                #print(shear_exp)
                P_exp = extrap_tests['exp'][2][~np.isnan((extrap_tests['exp'][1]))]
                P_exp = P_scaler.transform(np.array(P_exp).reshape(-1,1))
            except:
                print(f'Test failed for {sample}.')
                continue
            print(shear_exp.shape)
            if shear_exp.shape[0] < 5 and plot:
                continue
            os.makedirs(path + 'shear/HypNet', exist_ok = True)
            #The inputs for the ML model to predict
            print('M_ex_og', M_ex_og, np.power(10, M_ex_og))
            print('XX OG', XX_ex)
            XX_ex = scaler.transform((XX_ex))
            M_ex = M_scaler.transform(M_ex_og)
            S_ex = S_scaler.transform(np.log10(S_ex_og+0.00001))
            print('scaled S', S_ex)
            S_ex_torch_2 = y_scaler.transform(np.log10(S_ex_og+0.00001))
            #print(S_ex_og)
            #print(S_ex_torch_2)
            T_ex = T_scaler.transform(T_ex_og)
            P_ex = P_scaler.transform(P_ex_og)
            temps = [-50, -20, 20, 50]

            #start building figure
            fig = plt.figure(figsize = (7,10))
            fig.suptitle(f"HyperNet Prediction - {extrap_tests['sample']}")
            ax1 = plt.subplot(1,1,1)
            ax1.set_box_aspect(1)

            ax = [ax1]
            
            tick = 0

            #GPR Extrapolation
            l = shear_exp.shape[0]
            print('min gpr',np.argmin(gp_cv))
            test_pred, var = gpr_models_f[np.argmin(gp_cv)-1].predict_f_compiled(np.concatenate((XX_ex, M_ex, S_ex, T_ex, P_ex), axis = 1))
            test_pred = y_scaler.inverse_transform(np.array(test_pred))
            print('gpr var un_sc', var)
            var = y_scaler.inverse_transform(np.array(var).reshape(-1, 1)-1) - y_scaler.data_min_
            
            print('GPR OUT',test_pred, var)
            try:
                n_0, n, S_cr, r2  = fit_shear(np.array([a[0] for a in S_ex_og][:35]).astype(float).reshape(-1,), np.array([a for a in test_pred][:35]).astype(float).reshape(-1,))
                shear_const['GPR'] = pd.concat([shear_const['GPR'], pd.DataFrame({'Sample': [sample], 'z_shear': [n_0], 'S_cr': [S_cr], 'n':[n], 'r2':[r2]})], ignore_index = True)
            except:
                print(f'Could not fit sample {sample} with GPR')
                fitting_error['Shear']['GPR'] += 1
            
            line1 = ax[tick].plot(np.log10(S_ex_og).reshape(-1), test_pred.reshape(-1,), '--',label = f'GPR Prediction', color = colors['GPR'])
            plt.fill_between(np.log10(S_ex_og).reshape(-1), (test_pred - var).reshape(-1,), (test_pred + var).reshape(-1,), alpha = 0.2, color = colors['GPR'])

            #ANN Extrapolation
            test_pred, test_var,_ = predict_all_cv(models_f,[XX_ex, M_ex, S_ex, T_ex, P_ex])
            #exp_pred, _,_ = predict_all_cv(models_f[0],[XX_ex[:l], M_scaler.transform(np.array(Mw_ex).reshape(-1,1)), S_ex[:l], T_ex[:l], P_exp])
            #exp_pred =  y_scaler.inverse_transform(np.array(exp_pred).reshape(-1, 1)).reshape(-1,)
            test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
            test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)-1) - y_scaler.data_min_).reshape(-1,)
            try:
                n_0, n, S_cr, r2  = fit_shear(np.array([a[0] for a in S_ex_og][:35]).astype(float), np.array([a for a in test_pred][:35]).astype(float))
                shear_const['ANN'] = pd.concat([shear_const['ANN'], pd.DataFrame({'Sample': [sample], 'z_shear': [n_0], 'S_cr': [S_cr], 'n':[n], 'r2': [r2]})], ignore_index = True)
            except:
                fitting_error['Shear']['ANN'] += 1
                print(f'Could not fit sample {sample} with ANN')

            line1 = ax[tick].plot(np.log10(S_ex_og).reshape(-1), test_pred.reshape(-1,), '--',label = f'ANN Prediction',  color = colors['ANN'])
            plt.fill_between(np.log10(S_ex_og).reshape(-1), (test_pred - test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,), alpha = 0.2, color = colors['ANN'])
            
            
            #HypNet Exptrapolation
            # try:
            #     test_pred, test_var, _ = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True)
            #     test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
            #     test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)) - y_scaler.data_min_).reshape(-1,)
            #     try:
            #         n_0, n, S_cr  = fit_shear(np.array([a[0] for a in S_ex_og][:30]).astype(float), np.array([a for a in test_pred][:30]).astype(float))
            #         predicted_constants_HypNet = pd.concat([predicted_constants_HypNet, pd.DataFrame({'Sample': [sample], 'z_shear': [n_0], 'S_cr': [S_cr], 'n':[n]})], ignore_index = True)
            #     except:
            #         print(f'Could not fit sample {sample} with HyperNet')
            
            # except:
            #     print(f'HypNet Prediction Error on {sample}.')

            #try:
            print('PIMI inputs', XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex)
            test_pred, test_var, _ = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True)
            constants = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True, get_constants = True)
            test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
            test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)-1) - y_scaler.data_min_).reshape(-1,)
            pihn_pred = test_pred
            pihn_var = test_var
            eta_0_fit, n_fit, S_cr_fit, r2  = fit_shear(np.array([a[0] for a in S_ex_og]).astype(float), np.array([a for a in test_pred]).astype(float))
            n_0, n, S_cr = y_scaler.inverse_transform(constants['eta_0'].reshape(-1, 1))[0], constants['n'][0], y_scaler.inverse_transform(constants['tau'].reshape(-1, 1))[0]
            print(sample)
            print(r2)
            #print(np.array([a[0] for a in S_ex_og]).astype(float).shape)
            #print(test_pred.shape)
            
            #r2 = r2_score(test_pred, shear_func_obj(np.array([a[0] for a in S_ex_og]).astype(float), n_0, n, S_cr))
            shear_const['PIM'] = pd.concat([shear_const['PIM'], pd.DataFrame({'Sample': [sample], 'z_shear': [n_0], 'S_cr': [S_cr], 'n':[n], 'r2':[r2]})], ignore_index = True)
            # except:
            #     fitting_error['Shear']['PIMI'] += 1
            #     print(f'Could not fit shear sample {sample} with HyperNet')
            #exp_pred, _, _ = predict_all_cv(HypNet_f, [XX_ex[:l], y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex[:l])).reshape(-1,1)), y_scaler.transform(shear_exp.reshape(-1,1)), T_ex[:l], P_ex[:l]], is_torch = True)
            #exp_pred =  y_scaler.inverse_transform(np.array(exp_pred).reshape(-1, 1)).reshape(-1,)
            print(test_pred, type(test_pred))
            line1 = ax[tick].plot(np.log10(S_ex_og).reshape(-1), test_pred.reshape(-1,), '--',label = f'HypNet Prediction', color = colors['PIHN'])
            plt.fill_between(np.log10(S_ex_og).reshape(-1,), (test_pred - test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,), alpha = 0.2, color = colors['PIHN'])
            ax[tick].set_ylim(2, 9)
            ax[tick].set_xticks(list(np.arange(-4,8,2)),[r'$10^{-4}$', r'$10^{-2}$'] + [rf'$10^{i}$' for i in list(np.arange(0,8,2))])
            ax[tick].set_yticks(list(np.arange(2,9,2)), [rf'$10^{i}$' for i in list(np.arange(2,9,2))])
            ax[tick].set_ylabel(r'$\eta$ (Poise)')
            ax[tick].set_xlabel(r'$\dot{\gamma}$ (1/s)')
            exp = ax[tick].scatter(np.log10(shear_exp), log_visc_ex, c = 'orange', label = f'Experimental at {T_ex_og[0][0]} C')
            #ax[tick].plot(S_scaler.inverse_transform(S_ex), shear_func_obj(np.array([a[0] for a in S_ex_og]).astype(float), eta_0_fit, n_fit, S_cr_fit))
            plt.legend()
            #plt.savefig(f"../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/shear/{extrap_tests['sample']}_{sid}.svg")
            plt.savefig(f"{save_path}/{date}_{data_type}/shear/HypNet/{extrap_tests['sample']}_{M_ex_og[0]}_{T_ex_og[0]}_{sid}.png")
            plt.savefig(f"{save_path}/{date}_{data_type}/shear/HypNet/{extrap_tests['sample']}_{M_ex_og[0]}_{T_ex_og[0]}_{sid}.svg", dpi = 300)



        print('Saved shear extrap files')

    if DO_TEMP:
        
        print('Starting temperature extrapolations...')
        ###Temperature
        for tid in temp_ids:#[3084.0]:
            for r in t_ranges:
                
                extrap_tests = temp_test(temp_samps, tid, temp_range = r)
                XX_ex, M_ex_og,S_ex_og,T_ex_og, P_ex_og = extrap_tests['data_in']
                sample = extrap_tests['sample']
                log_visc_ex = extrap_tests['exp'][1][~np.isnan(np.power(10,extrap_tests['exp'][1]))]
                temp_exp = extrap_tests['exp'][0][~np.isnan((extrap_tests['exp'][1]))]
                if temp_exp.shape[0] < 3 and plot:
                   continue
                os.makedirs(path + 'temp/test', exist_ok = True)
                os.makedirs(path + f'temp/test/{r}', exist_ok = True)
                print(sample)
                P_exp = extrap_tests['exp'][2][~np.isnan((extrap_tests['exp'][1]))]
                P_exp = P_scaler.transform(np.array(P_exp).reshape(-1,1))
                print('M_ex_og_TEMP', M_ex_og, 'S_ex_og_TEMP', S_ex_og, 'T_ex_og_TEMP', T_ex_og)
                #The inputs for the ML model to predict
                XX_ex = scaler.transform((XX_ex))
                M_ex = M_scaler.transform(M_ex_og)
                S_ex = S_scaler.transform(np.log10(S_ex_og+0.00001))
                S_ex_torch_1 = S_torch_scaler.transform(S_ex_og)
                S_ex_torch_2 = y_scaler.transform(np.log10(S_ex_og+0.00001))

                T_ex = T_scaler.transform(T_ex_og)
                P_ex = P_scaler.transform(P_ex_og)

                #start building figure
                fig = plt.figure(figsize = (7,10))
                ax1 = plt.subplot(1,1,1)
                ax1.set_box_aspect(1)
                ax = [ax1]
                tick = 0

                #GPR Extrapolation
                l = temp_exp.shape[0]
                test_pred, var = gpr_models_f[np.argmin(gp_cv)-1].predict_f_compiled(np.concatenate((XX_ex, M_ex, S_ex, T_ex, P_ex), axis = 1))
                print('GPR out', test_pred)
                print('GPR var', var)
                
                test_pred = y_scaler.inverse_transform(np.array(test_pred))
                #var = np.array((y_scaler.data_max_ - y_scaler.data_min_)*var).reshape(-1,1)
                var = np.array(var)
                # print(np.array([a[0]+ 273.15 for a in T_ex_og], dtype =float))
                # test = WLF_obj(np.array([a[0]+ 273.15 for a in T_ex_og], dtype= float), 300, 17, 52, 5)
                # print(test)
                try:
                    try:
                        to_fit = np.array([a for a in test_pred]).astype(float).reshape(-1,)
                        Tr, C1, C2, eta_r, r2 = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), to_fit, sigma = var.reshape(-1,))
                        WLF_const['GPR'] = pd.concat([WLF_const['GPR'], pd.DataFrame({'Sample': extrap_tests['sample'], 'C1': [C1], 'C2':[C2], 'Tr':[Tr], 'r2':[r2], 'r':[r]})], ignore_index= True)
                    except:
                        Tr, C1, C2, eta_r, r2 = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), to_fit + np.random.rand(*to_fit.shape), sigma = var.reshape(-1,))
                        WLF_const['GPR'] = pd.concat([WLF_const['GPR'], pd.DataFrame({'Sample': extrap_tests['sample'], 'C1': [C1], 'C2':[C2], 'Tr':[Tr], 'r2':[r2], 'r':[r]})], ignore_index= True)
                except:
                    fitting_error['Temperature']['GPR'] += 1
                    print(f'GPR failed for temperature {sample}')

                line1 = ax[tick].plot(T_ex_og.reshape(-1,), test_pred.reshape(-1,), '--',label = f'GPR Prediction', color = colors['GPR'])
                plt.fill_between(T_ex_og.reshape(-1,), (test_pred - var).reshape(-1,) , (test_pred + var).reshape(-1,), alpha = 0.2, color = colors['GPR'])

                #ANN Extrapolation
                test_pred, test_var,_ = predict_all_cv(models_f,[XX_ex, M_ex, S_ex, T_ex, P_ex])
                
                test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
                test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)-1) - y_scaler.data_min_).reshape(-1,)
                try: 
                    try:
                        to_fit = np.array([a for a in test_pred]).astype(float)
                        Tr, C1, C2, eta_r, r2 = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float), to_fit, sigma = test_var.reshape(-1,))
                        WLF_const['ANN'] = pd.concat([WLF_const['ANN'], pd.DataFrame({'Sample': extrap_tests['sample'], 'C1': [C1], 'C2':[C2], 'Tr':[Tr], 'r2':[r2], 'r':[r]})], ignore_index= True)
                    except:
                        Tr, C1, C2, eta_r, r2 = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float), to_fit + np.random.rand(*to_fit.shape), sigma = test_var.reshape(-1,))
                        WLF_const['ANN'] = pd.concat([WLF_const['ANN'], pd.DataFrame({'Sample': extrap_tests['sample'], 'C1': [C1], 'C2':[C2], 'Tr':[Tr], 'r2':[r2], 'r':[r]})], ignore_index= True)
                except:
                    fitting_error['Temperature']['ANN'] += 1
                    print(f'ANN failed for temperature {sample}')
                line1 = ax[tick].plot(T_ex_og.reshape(-1,), test_pred.reshape(-1,), '--',label = f'ANN Prediction',  color = colors['ANN'])
                plt.fill_between(T_ex_og.reshape(-1,), (test_pred - test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,), alpha = 0.2, color = colors['ANN'])
                
                #HypNet Exptrapolation
                # try:
                if any(np.isnan(y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)))):
                    print('invalid mol weights')
                    print(M_ex)
                    continue
                if any(np.isnan(S_ex_torch_2)):
                    print('invalid shear rates')
                    print(S_ex_torch_2)
                    continue

                test_pred, test_var, _ = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True)
                test_pred = y_scaler.inverse_transform(np.array(test_pred).reshape(-1, 1)).reshape(-1,)
                constants = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, T_ex, P_ex], is_torch = True, get_constants = True)
                test_var = (y_scaler.inverse_transform(np.array(test_var).reshape(-1, 1)-1) - y_scaler.data_min_).reshape(-1,)
                
                #try:
                #Tr, C1, C2 = T_scaler.inverse_transform(constants["Tr"][0].reshape(-1, 1)), y_scaler.inverse_transform(constants["c1"][0].reshape(-1, 1)), T_scaler.inverse_transform(constants["c2"][0].reshape(-1, 1))
                #print(T_ex.shape)
                C1_adj = C1 - y_scaler.data_min_
                Tr_y, _, _ = predict_all_cv(HypNet_f,[XX_ex, y_scaler.transform(M_scaler.inverse_transform(np.array(M_ex)).reshape(-1,1)), S_ex_torch_2, np.ones_like(T_ex)*constants["Tr"][0], P_ex], is_torch = True)
                print('Tr_unsc', np.mean(constants["Tr"]).reshape(-1, 1), 'Tr_y', Tr_y, 'C1_unsc', np.mean(constants["c1"]).reshape(-1, 1), 'C2_unsc', np.mean(constants["c2"]).reshape(-1, 1))
                
                unsc_const = [np.mean(constants["Tr"]).reshape(-1, 1), np.array(constants["c1"]).reshape(-1, 1), np.mean(constants["c2"]).reshape(-1, 1), Tr_y[0]]
                unsc_wlf = y_scaler.inverse_transform(WLF_obj(T_ex, *unsc_const).reshape(-1,1)).reshape(-1)
                Tr, C1, C2, Tr_y, r2_sc = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), unsc_wlf)
                
                #Tr_y = y_scaler.inverse_transform(Tr_y[0].reshape(-1, 1))
                
                #print('testpred', test_pred)
                Tr_fit, C1_fit, C2_fit, eta_r_fit, r2_fit = fit_WLF(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), np.array([a for a in test_pred]).astype(float).reshape(-1,), sigma = test_var.reshape(-1,))
            # C2_adj = C2 - T_scaler.data_min_
                wlf_pihn_fit = WLF_obj(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), Tr , C1, C2, Tr_y).reshape(-1,)
                print(wlf_pihn_fit.shape,wlf_pihn_fit )
                print(test_pred.shape, test_pred)
                r2 = r2_score(test_pred.reshape(-1,), wlf_pihn_fit)


                print('Tr', Tr, 'Tr_y', Tr_y, 'C1',C1, 'C1', C1_adj, 'C2', C2, 'r2 of scaling', r2_sc)
                print('Tr_fit', Tr_fit, 'Tr_y_fit', eta_r_fit, 'C1_fit',C1_fit, 'C2_fit', C2_fit)
            
                print('r2', r2, r2_fit)
                WLF_const['PIM'] = pd.concat([WLF_const['PIM'], pd.DataFrame({'Sample': extrap_tests['sample'], 'C1': [C1], 'C2':[C2], 'Tr':[Tr], 'r2':[r2], 'r2_fit':[r2_fit], 'r':[r]})], ignore_index= True)
                #except:
                #    print(f"HypNet did not fit WLF for {extrap_tests['sample']}")
                if plot:
                    #Plot for temperature
                    line1 = ax[tick].plot(T_ex_og.reshape(-1,), test_pred.reshape(-1,), '--',label = f'HypNet Prediction', color = colors['PIHN'])
                    plt.fill_between(T_ex_og.reshape(-1,), (test_pred - test_var).reshape(-1,) , (test_pred + test_var).reshape(-1,), alpha = 0.2, color = colors['PIHN'])
                    ax[tick].set_ylabel(r'$\eta$ (Poise)')
                    ax[tick].set_xlabel(r'Temperature (C)')
                    ax[tick].set_ylim(-2,6)
                    exp = ax[tick].scatter(temp_exp, log_visc_ex, c = 'orange', label = f'Experimental at {T_ex_og[0][0]} C')
                    #ax[tick].set_yticks(list(np.arange(0,3,1)), [rf'$10^{i}$' for i in list(np.arange(0,3,1))])
                    #ax[tick].plot(T_ex_og.reshape(-1,), wlf_pihn_fit)
                    #ax[tick].plot(T_ex_og.reshape(-1,), WLF_obj(np.array([a[0] + 273.15 for a in T_ex_og]).astype(float).reshape(-1,), Tr_fit, C1_fit, C2_fit, eta_r_fit).reshape(-1,), color = 'r')
                    #ax[tick].text(T_ex_og[floor(T_ex_og.shape[0]/4)], wlf_pihn_fit[-1], r'$r2=$' + "{:1.4f}".format(r2), fontsize = 13)
                    #ax[tick].text(T_ex_og[floor(T_ex_og.shape[0]/4)], wlf_pihn_fit[-1] + 0.5, r'$r2_fit=$' + "{:1.4f}".format(r2_fit), fontsize = 13)
                    ax[tick].set_yticks(list(np.arange(-2,6,2)), [r'$10^{-2}$'] + [rf'$10^{i}$' for i in list(np.arange(0,6,2))])
                    
                    
                    plt.legend()
                        #plt.savefig(f"../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/temp/{extrap_tests['sample']}_{M_ex_og[0]}_{S_ex_og[0]}_{tid}.svg")
                    
                    plt.savefig(f"{save_path}/{date}_{data_type}/temp/test/{r}/{extrap_tests['sample']}_{S_ex_og[0]}_{M_ex_og[0]}_{tid}.png")
                    plt.savefig(f"{save_path}/{date}_{data_type}/temp/test/{r}/{extrap_tests['sample']}_{S_ex_og[0]}_{M_ex_og[0]}_{tid}.svg")
                plt.close()

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/gpr_WLF_const.pickle', 'wb') as handle:
        #     pickle.dump(WLF_constants_gpr, handle)

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/ANN_WLF_const.pickle', 'wb') as handle:
        #     pickle.dump(WLF_constants_ann, handle)

        # with open(f'../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/{date}_{data_type}/HypNet_WLF_const.pickle', 'wb') as handle:
        #     pickle.dump(WLF_constants_HypNet, handle)
    data['Model_num'] = models_run
    data['Model'] = f'{date}_{data_type}'
    if models_run == 0:
        predicted_data = data
    else:
        predicted_data = pd.concat([predicted_data, data])

    with open(f'{save_path}/Mw_const.pickle', 'wb') as handle:
        pickle.dump(Mw_const, handle)

    with open(f'{save_path}/shear_const.pickle', 'wb') as handle:
        pickle.dump(shear_const, handle)

    with open(f'{save_path}/WLF_const.pickle', 'wb') as handle:
        pickle.dump(WLF_const, handle)
    models_run += 1
    print('Finished extrapolation tests')


if __name__ == '__main__':
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
    test_folder = '../../../data/ayush/Melt_Viscosity_Predictor/Extrapolation_Tests/0307_polytrial'
    os.makedirs(test_folder, exist_ok= True)
    models_list = os.listdir('./MODELS')
    #models_list = ['2023-01-03_polysplitnew_0_polybutane13diolaltsebacicacid_t2']
    models_list = [m for m in models_list if 'polysplitnew' in m and '4-methyl-1-pentene' in m]
    print('total_num_models', len(models_list))
    tot_num_models = len(models_list)
    date = [m[:10] for m in models_list]
    data_type = [m[11:] for m in models_list]
    date = date
    data_type = data_type
    print(date)
    print(data_type)
    for d, dt in zip(date, data_type):
        print(d, dt)
        if os.path.exists(f'MODELS/{d}_{dt}/test_evals.pkl'):
            if os.path.exists(f'{test_folder}/{d}_{dt}'):
                models_run += 1
                continue    
            try:
                main(d, dt, test_folder, plot = True)
            except ModelLoadingError:
                continue
        else:
            tot_num_models -= 1
            print(f'Skipped {d}_{dt}.')
        print(f'Last completed model = {d}_{dt}')
    
    print(f'{models_run} / {tot_num_models} models evaluated')
    print(f'{samples_run} / {1903} samples extrapolated')
    print(model_loading_errors)
    predicted_data.to_excel(f'{test_folder}/predicted_data.xlsx')
    fitting_error.to_excel(f'{test_folder}/fitting_error.xlsx')
    with open(f'{test_folder}/Mw_const.pickle', 'wb') as handle:
        pickle.dump(Mw_const, handle)

    with open(f'{test_folder}/shear_const.pickle', 'wb') as handle:
        pickle.dump(shear_const, handle)

    with open(f'{test_folder}/WLF_const.pickle', 'wb') as handle:
        pickle.dump(WLF_const, handle)