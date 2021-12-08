from sklearn.model_selection import KFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def viscNN_LC(create_model, X_tot, Y_tot, logMw, log_shear, Temp,S_scaler=None, T_scaler=None, M_scaler= None):
    tr_sizes = list(np.linspace(0.10, 0.90, 9))
    train_error = []
    test_error = []
    for size in tr_sizes:
        XX, X_test, yy, y_test, M, M_test, S, S_test, T, T_test = train_test_split(X_tot, Y_tot, logMw, log_shear, Temp, test_size= 1 - size)
        model = crossval_NN(create_model, XX, yy, M, S, T, S_scaler= S_scaler, T_scaler=T_scaler,M_scaler= M_scaler)
        if S_scaler is None:
            fit_in = [XX, M, S, T]
            eval_in = [X_test, M_test, S_test, T_test]
        else:
            fit_in = [XX, M, S, T, M_scaler.inverse_transform(M), S_scaler.inverse_transform(S), T_scaler.inverse_transform(T)]
            eval_in = [X_test, M_test, S_test, T_test, M_scaler.inverse_transform(M_test), S_scaler.inverse_transform(S_test), T_scaler.inverse_transform(T_test)]
        train_error.append(model.evaluate(fit_in, yy, verbose = 0))
        test_error.append(model.evaluate(eval_in, y_test, verbose = 0))

    return tr_sizes, train_error, test_error

def crossval_NN(create_model, XX, yy, M, S, T, S_scaler=None, T_scaler=None, M_scaler= None, verbose = 0, random_state = None):
    kf = KFold(n_splits=5, shuffle = True, random_state = random_state)
    m = []
    error = []
    for train_index, test_index in kf.split(XX):
        X_train, X_val = XX[train_index], XX[test_index]
        y_train, y_val = yy[train_index], yy[test_index]
        M_train, M_val = M[train_index], M[test_index]
        S_train, S_val = S[train_index], S[test_index]
        T_train, T_val = T[train_index], T[test_index]
        n_features = X_train.shape[1]
        m.append(create_model(n_features))
        if S_scaler is None:
            fit_in = [X_train, M_train, S_train, T_train]
            eval_in = [X_val, M_val, S_val, T_val]
        elif T_scaler is None:
            fit_in = [X_train, M_train, T_train, np.power(10, M_scaler.inverse_transform(M_train)), S_train]
            eval_in = [X_val, M_val, T_val, np.power(10, M_scaler.inverse_transform(M_val)), S_val]
        else:
            fit_in = [X_train, M_train, S_train, T_train, M_scaler.inverse_transform(M_train), S_scaler.inverse_transform(S_train), T_scaler.inverse_transform(T_train)]
            eval_in = [X_val, M_val, S_val, T_val, M_scaler.inverse_transform(M_val), S_scaler.inverse_transform(S_val), T_scaler.inverse_transform(T_val)]
        m[-1].fit(fit_in, y_train, epochs=300, batch_size=30, verbose=0)
        error.append(m[-1].evaluate(eval_in, y_val, verbose=0))
        if verbose > 0:
            print('MSE: %.3f, RMSE: %.3f' % (error[-1], np.sqrt(error[-1])))
    if verbose > 0:
        print('CV MSE error:' + str(np.mean(error)))
    return m[error.index(min(error))]

def delete_outlier(model, y_pred, y_test, TH = 12):
    ind = []
    for i in range(y_pred.shape[0]):
        if y_pred[i] > TH:
            y_pred = np.delete(y_pred, i, 0)
            y_test = np.delete(y_test, i, 0)
            ind.append(i)
            break

    return y_pred, y_test, ind

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
    zero_shear_data = data[data['SAMPLE_ID'].isin(z_shear_samples)]
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

def Mw_test(samples_df: pd.DataFrame, samps):
    out = []
    fp_cols = []
    for c in samples_df.columns:
        if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
            fp_cols.append(c)
    logMw = pd.Series(np.linspace(1,10,10))
    for i in samps:
        trial = pd.DataFrame(samples_df.loc[[i]])
        trial.head()
        fp = trial[fp_cols + ['SMILES']]
        tests = pd.DataFrame()
        tests['logMw'] = logMw
        tests['Temperature'] = trial['Temperature'].values[0]
        tests['Shear_Rate'] = trial['Shear_Rate'].values[0]
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

        if i == 2:
            print(trial)
        if trial['Log_K1'].values[0] != 0:
            tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] <= np.log10(trial['Mcr'].values[0]), 'logMw'] + trial['Log_K1'].values[0]
            tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'Melt_Viscosity'] = tests.loc[tests['logMw'] > np.log10(trial['Mcr'].values[0]), 'logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]
        else:
            tests['Melt_Viscosity'] = tests['logMw']*trial['Alpha'].values[0] + trial['Log_K2'].values[0]

        y = np.array(tests['Melt_Viscosity']).reshape(-1, 1)
        out.append({'tests':tests, 'data_in':[XX, OH_shear,M,S,T,y]})

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

    print(test_std)

    test_df['BAD_PRED'] = test_std < test_df['Error']
    train_df['BAD_PRED'] = train_std < train_df['Error']

    return train_df, test_df