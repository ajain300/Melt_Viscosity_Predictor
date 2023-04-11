from logging import raiseExceptions
from sklearn.model_selection import KFold, StratifiedKFold
from random import sample
import numpy as np
import pandas as pd
from .metrics import MSE, OME
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras_tuner as kt
import json
from os import sys
import os
sys.path.append('../')
from ViscNN import ViscNN_concat_HP
#from train_torch import run_training, MVDataset
from torch.utils.data.dataloader import DataLoader
from math import floor
import yaml
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
import torch

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
    # for c in samples_df.columns:
    #     if 'afp' in c or 'bfp' in c or 'mfp' in c or 'efp' in c:
    #         fp_cols.append(c)
    for c in samples_df.columns:
        if 'fp' in c:
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
    trial_shear = np.array(trial['Shear_Rate']).reshape(-1,1)
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
        # for i in sample_id:
        #     if len(data.loc[data['SAMPLE_ID'] == i]) <= 3:
        #         data = data.drop(data.loc[data['SAMPLE_ID'] == i].index, axis = 0)
        # sample_id = list(data.agg({'SAMPLE_ID': 'unique'})[0])

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
    print('Mw input to temp_test',trial['Mw'].values[0])
    tests['logMw'] = trial['Mw'].values[0]
    tests['Shear_Rate'] = trial['Shear_Rate'].values[0]
    tests['Polymer'] = trial['Polymer'].values[0]
    tests['PDI'] = trial['PDI'].values[0]
    tests.loc[[i for i in tests.index], fp_cols] = np.array(fp)
    #tests.loc[[i for i in tests.index], fp_cols + ['SMILES']] = np.array(fp)

    if trial['Shear_Rate'].values[0] != 0:
        #tests['Shear_Rate'] = np.log10(tests['Shear_Rate'])
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
    out = {'exp': [temp_exp, visc_exp, P_exp], 'data_in':[XX,M,S,T,P], 'sample': f'{trial[p].values[0]}'}

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