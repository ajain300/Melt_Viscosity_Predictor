from sklearn.model_selection import KFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
from validation.metrics import MSE, OME
import matplotlib.pyplot as plt
from os import sys
sys.path.append('../')
from MODELS.ViscNN import predict_all_cv

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


def crossval_compare(NN_models, XX, yy, M, S, T, S_trans, S_scaler, M_scaler, verbose = 1, random_state = None, epochs = 500, gpr_model = None):
    kf = KFold(n_splits=10, shuffle = True, random_state = random_state)
    NN = [[] for i in range(len(NN_models))]
    gpr = []
    gp_cv = []
    hist = [[] for i in range(len(NN_models))]
    for train_index, test_index in kf.split(XX):
        X_train, X_val = XX[train_index], XX[test_index]
        y_train, y_val = yy[train_index], yy[test_index]
        M_train, M_val = M[train_index], M[test_index]
        S_train, S_val = S[train_index], S[test_index]
        T_train, T_val = T[train_index], T[test_index]
        n_features = X_train.shape[1]
        for i in range(len(NN_models)): NN[i].append(NN_models[i](n_features)) #S_trans.inverse_transform(S_scaler.inverse_transform(S_train))
        fit_in = [X_train, M_train, S_train, T_train]
        eval_in = [X_val, M_val, S_val, T_val]
        for i in range(len(NN_models)): hist[i].append(NN[i][-1].fit(fit_in, y_train, epochs=epochs, batch_size=20, validation_data = (eval_in, y_val) ,verbose=0))
        if gpr_model:
            X_train_ = np.concatenate((X_train, M_train, S_train, T_train), axis = 1)
            X_test_ = np.concatenate((X_val, M_val, S_val, T_val), axis = 1)
            gpr.append(gpr_model(X_train_, y_train))
            m = gpr[-1]
            mean, var = m.predict_y(X_test_)
            gp_cv.append(OME(mean, y_val))
        if verbose > 0:
            #print('MSE: %.3f, RMSE: %.3f' % (error[-1], np.sqrt(error[-1])))
            print('Trained fold ' + str(len(hist[-1])) + ' ...')
            for i in range(len(NN_models)):
                print('CV Error ' + NN_models[i].__name__ + ': ' + str(hist[i][-1].history['val_loss'][-1]))
            if gpr_model: print('CV Error GPR: ' + str(gp_cv[-1]))
    #if verbose > 0:
        #print('CV MSE error:' + str(np.mean(error)))
    return NN, hist, gpr, gp_cv

def delete_outlier(model, y_pred, y_test, TH = 12):
    ind = []
    for i in range(y_pred.shape[0]):
        if y_pred[i] > TH:
            y_pred = np.delete(y_pred, i, 0)
            y_test = np.delete(y_test, i, 0)
            ind.append(i)
            break

    return y_pred, y_test, ind

def get_Mw_samples(data:pd.DataFrame):
    id = 0
    fp_cols = []
    for c in data.columns:
        if 'fp' in c:
            fp_cols.append(c)
    data = data.loc[data['Shear_Rate'] == 0 ]
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

    for c in ['Mw', 'Melt_Viscosity']:
        data[c] = np.log10(data[c])
    
    sample_id = list(data.agg({'SAMPLE_ID': 'unique'}))
    print(sample_id)

    for i in sample_id:
        if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
            #print()
            data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
            sample_id.remove(i)
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
    logMw = pd.Series(np.linspace(2,6,40))
    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)
    print(trial)
    fp = trial.loc[0,fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['logMw'] = logMw
    tests['Temperature'] = trial['Temperature'].values[0]
    tests['Shear_Rate'] = trial['Shear_Rate'].values[0]
    tests['Polymer'] = trial['Polymer'].values[0]
    tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)

    if trial['Shear_Rate'].values[0] != 0:
        tests['Shear_Rate'] = np.log10(test['Shear_Rate'])
        tests['SHEAR'] = 1
        tests['ZERO_SHEAR'] = 0
    else:
        tests['SHEAR'] = 0
        tests['ZERO_SHEAR'] = 1

    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    # if trial['Log_K1'].values[0] != 0:
    #     tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'logMw'] + trial['Log_K1'].values[0]
    #     tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
    # else:
    #     tests['Melt_Viscosity'] = tests['logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]

    Mw_exp = trial['Mw']
    visc_exp = trial['Melt_Viscosity']
    out = {'exp': [Mw_exp, visc_exp], 'data_in':[XX, OH_shear,M,S,T], }

    return out

def get_shear_samples(data:pd.DataFrame):
    """
    Takes in overall data and extrapolates zero-shear Mw values from samples
    """
    fp_cols = []
    for c in data.columns:
        if 'fp' in c:
            fp_cols.append(c)
    id = 0
    data = data.loc[data['Shear_Rate'] != 0]
    temp = data.loc[data.index[0], 'Temperature']
    poly = data.loc[data.index[0], 'Polymer']
    weight = data.loc[data.index[0], 'Mw']
    fp = data.loc[data.index[0], fp_cols]

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

    for c in ['Mw', 'Melt_Viscosity']:
        data[c] = np.log10(data[c])
    
    sample_id = list(data.agg({'SAMPLE_ID': 'unique'}))
    for i in sample_id:
        if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
            #print()
            data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
            sample_id.remove(i)
    return data, sample_id

def shear_test(samples_df: pd.DataFrame, samp):
    out = []
    fp_cols = []
    for c in samples_df.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            fp_cols.append(c)
    log_shear = pd.Series(np.linspace(-3,7,30))
    trial = pd.DataFrame(samples_df.loc[samples_df['SAMPLE_ID'] == samp]).reset_index(drop = True)
    print(trial)
    fp = trial.loc[0, fp_cols + ['SMILES']]
    tests = pd.DataFrame()
    tests['Shear_Rate'] = np.power(10, log_shear)
    tests['logMw'] = trial.loc[0,'Mw']
    tests['Temperature'] = trial.loc[0,'Temperature']
    tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)


    XX = tests[fp_cols]
    M = np.array(tests['logMw']).reshape(-1, 1)
    S = np.array(tests['Shear_Rate']).reshape(-1, 1)
    T = np.array(tests['Temperature']).reshape(-1, 1)
    #OH_shear = np.array(tests[['SHEAR', 'ZERO_SHEAR']])

    y = np.array(trial['Melt_Viscosity']).reshape(-1, 1)
    trial_shear = np.log10(np.array(trial['Shear_Rate'])).reshape(-1,1)
    out = {'known':[trial_shear, y], 'data_in':[XX,M,S,T]}

    return out


def evaluate_model(Y_test, Y_train, filtered_data, ind):
    test_df = filtered_data.iloc[ind[1],:]
    test_df['Y_pred'] = Y_test
    train_df = filtered_data.iloc[ind[0],:]
    train_df['Y_pred'] = Y_train

    test_df['Error'] = abs(test_df['Melt_Viscosity'] - test_df['Y_pred'])#/test_df['Melt_Viscosity']
    train_df['Error'] = abs(train_df['Melt_Viscosity'] - train_df['Y_pred'])#/train_df['Melt_Viscosity']

    test_std = np.std(test_df['Error'])
    train_std = np.std(train_df['Error'])

    print(train_std)

    test_df['BAD_PRED'] = test_std < test_df['Error']
    train_df['BAD_PRED'] = train_std < train_df['Error']

    return train_df, test_df
