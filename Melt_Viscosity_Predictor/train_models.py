from copyreg import pickle
from posixpath import split
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from utils.metrics import OME, MSE, get_CV_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from ViscNN import create_ViscNN_concat
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data.dataloader import DataLoader
from utils.model_utils import predict_all_cv, load_models
from utils.train_utils import *
from model_versions import *
from Visc_PENN import *
import keras_tuner as kt
#import gpflow
#from gpflow_tools.gpr_models import train_GPR, create_GPR, GPflowRegressor
from data_tools.dim_red import fp_PCA
from data_tools.data_viz import val_epochs
import datetime
import keras.backend as K
import random
import yaml
import json
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
import os
import time
import joblib
from utils.Gpuwait import GpuWait
from typing import Dict
from pgfingerprinting import fp
import ast
import seaborn as sns
import matplotlib.gridspec as gridspec
import multiprocessing
import traceback
import pickle

parser = argparse.ArgumentParser(description='Get training vars')
parser.add_argument('--config', default='./config.yaml', help = 'get config file')

# Paths
TRAINING_DIR = "/data/ayush/Melt_Viscosity_Predictor/Melt_Viscosity_Predictor/MODELS/"
DATASET_DIR = "data_splits"

class ModelTrainer:
    fp_memo = {}
    def __init__(self, name : str, config_args : dict, models : list[BaseModel], data_split_type = 'polyphysics'):
        # Set training directory
        self.name = name
        os.makedirs(os.path.join(TRAINING_DIR, name), exist_ok=True)
        self.sim_dir = os.path.join(TRAINING_DIR, name)
        self.args = config_args
        self.models = models
        #USE GPU
        # with GpuWait(10, 3600*10, 0.9) as wait:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Read data
        data = self._get_data()
 
        if not self._data_split_exists():
            # Fingerprint data with pg fingerprinting
            self.full_data_fp = self._fingerprint_dataset(data)

            # Get test train split
            self.full_data = data
            self.train_df, self.test_df, self.test_samples_unselect = self.split_data(data, data_split_type)
            self.train_df_fp, self.test_df_fp = self.full_data_fp.loc[self.train_df.index], self.full_data_fp.loc[self.test_df.index]
            
            # Plot the data split
            self.plot_data_split()

            # Save the dataframes
            self.save_data_split()
        else: 
            print("Loading data from folder.")
            self.load_data_split()

        # Create scalers
        self.scalers = self.data_scalers(self.train_df, self.train_df_fp)
        
    def _get_data(self):
        data = pd.read_excel(self.args.file)
        data.columns = [str(c) for c in data.columns]
        if args.aug == True and 'level_0' in data.columns:
            data = data.drop(columns = 'level_0')
        
        data = data[[FeatureHeaders.smiles.value,
                    FeatureHeaders.mol_weight.value,
                    FeatureHeaders.shear_rate.value,
                    FeatureHeaders.PDI.value,
                    FeatureHeaders.temp.value,
                    FeatureHeaders.visc.value,
                    FeatureHeaders.sample_type.value,
                    "Weight 1", "Weight 2"]]
        

        # Check data for needed features and delete the rows with nans
        check_df = data[[FeatureHeaders.smiles.value,
                    FeatureHeaders.mol_weight.value,
                    FeatureHeaders.shear_rate.value,
                    FeatureHeaders.PDI.value,
                    FeatureHeaders.temp.value,
                    FeatureHeaders.visc.value,
                    FeatureHeaders.sample_type.value]]
        nan_rows = check_df[check_df.isna().any(axis=1)].index
        data = data.drop(nan_rows)

        for c in ['Mw', 'Melt_Viscosity']:
            data[c] = np.log10(data[c])

        # convert smiles column into a proper list
        data[FeatureHeaders.smiles.value] = data[FeatureHeaders.smiles.value].apply(convert_smiles_list)

        # convert temp column from C to K
        data[FeatureHeaders.temp.value] = data[FeatureHeaders.temp.value]+273.15

        # Determine shear type and impute missing PDI values
        data['ZERO_SHEAR'] = 1
        data['SHEAR'] = 0
        for i in data.index:
            if data.loc[i, 'Shear_Rate'] != 0:
                data.loc[i, 'SHEAR'] = 1
                data.loc[i, 'ZERO_SHEAR'] = 0
            if not data.loc[i,'PDI'] > 0 or data.loc[i,'PDI'] > 100:
                data.loc[i,'PDI'] = 2.06

        return data

    def split_data(self, df : pd.DataFrame, split_type : str):
        os.makedirs(os.path.join(TRAINING_DIR, self.name, DATASET_DIR), exist_ok=True)
        if split_type == 'polyphysics':
            train_df, test_df, tested_samples_unselected_df,split_conditions, group_median = poly_physics_train_test_split_unique_median(df, "SMILES", 
            [FeatureHeaders.mol_weight.value])
            group_median.to_csv(os.path.join(self.sim_dir, "group_medians.csv"))
        elif split_type == 'poly_N_exp':
            train_df, test_df = poly_physics_train_test_split_N_train(df, "SMILES", 5)
        else:
            raise Exception("Incorrect data split type.")
        


        return train_df, test_df, tested_samples_unselected_df

    def data_scalers(self, df : pd.DataFrame, fp_df : pd.DataFrame) -> dict:
        """
        Create scalers for all features.
        """
        scalers: Dict[str, MinMaxScaler] = {}
        shear = np.array(df[FeatureHeaders.shear_rate.value], dtype = float).reshape((-1,1))
        Temp = np.array(df[FeatureHeaders.temp.value]).reshape((-1,1))
        PDI = np.array(df[FeatureHeaders.PDI.value]).reshape((-1,1))
        scalers[FeatureHeaders.fp.value] = MinMaxScaler(copy = False, feature_range = (-1,1)).fit(fp_df)
        #XX = np.array(scaler.fit(df.filter(fp_cols)).transform(df.filter(fp_cols)))
        
        yy = np.array(df.loc[:,'Melt_Viscosity']).reshape((-1,1))
        scalers[FeatureHeaders.visc.value] = MinMaxScaler(feature_range = (-1,1)).fit(yy)
        scalers[FeatureHeaders.temp.value] = MinMaxScaler(feature_range = (-1,1)).fit(Temp)
        scalers[FeatureHeaders.inv_temp.value] = MinMaxScaler(feature_range = (-1,1)).fit(1/Temp)
        logMw = np.array(df[FeatureHeaders.mol_weight.value], dtype = float).reshape((-1,1))
        scalers[FeatureHeaders.mol_weight.value] = MinMaxScaler(feature_range = (-1,1)).fit(logMw)
        scalers[FeatureHeaders.shear_rate.value] = MinMaxScaler(feature_range = (-1,1)).fit(np.log10(shear + 0.00001))
        scalers[FeatureHeaders.PDI.value] = MinMaxScaler(feature_range = (-1,1)).fit(np.log(PDI))
        

        scaler_path = os.path.join(self.sim_dir, "dataset", "scalers.pkl")
        # save scalers
        with open(scaler_path, 'wb') as file:
            # Dump the dictionary to the file using pickle
            pickle.dump(scalers, file)
        return scalers
    
    def _fingerprint_copolymer(self, smiles_list, conc_list) -> dict:
        """
        Given the SMILES of a blend, calculate the fingerprint.
        TODO assumed a blend with two homopolymers
        """
        
        # Get SMILES for all homopolymers and collect in df
        fp_df_list = []
        for i, smiles in enumerate(smiles_list):
            if smiles in self.fp_memo.keys():
                homo_fp = self.fp_memo[smiles]
            else:
                homo_fp = fp.fingerprint_from_smiles(smiles)
                self.fp_memo[smiles] = homo_fp
            fp_df_list.append(pd.DataFrame([homo_fp], index = [i]))

        # combine dfs
        combined_df = pd.concat(fp_df_list, axis=0).fillna(0)

        # use the dfs to calculate harmonic sum
        w_1 = conc_list[0]
        w_2 = conc_list[1]
        fp_1 = combined_df.loc[0]
        fp_2 = combined_df.loc[1]
        co_fp = w_1*fp_1 + w_2*fp_2

        return co_fp.to_dict()
        
    def _fingerprint_blend(self, smiles_list, conc_list) -> dict:
        """
        Given the SMILES of a blend, calculate the fingerprint.
        TODO assumed a blend with two homopolymers
        """
        
        # Get SMILES for all homopolymers and collect in df
        fp_df_list = []
        for i, smiles in enumerate(smiles_list):
            if smiles in self.fp_memo.keys():
                homo_fp = self.fp_memo[smiles]
            else:
                homo_fp = fp.fingerprint_from_smiles(smiles)
                self.fp_memo[smiles] = homo_fp
            fp_df_list.append(pd.DataFrame([homo_fp], index = [i]))

        # combine dfs
        combined_df = pd.concat(fp_df_list, axis=0).fillna(0)

        # use the dfs to calculate harmonic sum
        w_1 = conc_list[0]
        w_2 = conc_list[1]
        fp_1 = combined_df.loc[0]
        fp_2 = combined_df.loc[1]
        blend_fp = 1/((w_2/(fp_1+1))+ (w_1/(fp_2+1))) - 1
        
        return blend_fp.to_dict()
    
    def _fingerprint_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Uses pgfingerprinting to evaluate the fingerprints for all samples in the dataset

        Args:
        data: Dataframe with smiles and weight values

        """
        # Fingerprint homopolymers, copolymers, and blends
        # TODO make this more efficient
        
        # If the fingerprint dataset already exists, use this instead of re-fingerprinting
        if os.path.exists(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl")):
            print(f"Read fingerprinted dataset from {os.path.join(self.sim_dir, 'dataset', 'full_data_fp.pkl')} ...")
            fp_df = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl"))
            return fp_df
        
        fp_dict = {}
        data = data.copy()
        
        print(f"Fingerprinting dataset...")
        start = time.time()
        for row in data.index:
            if data.loc[row, FeatureHeaders.sample_type.value] == 'Homopolymer':
                smiles = data.loc[row, FeatureHeaders.smiles.value][0]
                if smiles in self.fp_memo.keys():
                    fp_dict[row] = self.fp_memo[smiles]
                else:
                    fp_dict[row] = fp.fingerprint_from_smiles(smiles)
                    self.fp_memo[smiles] = fp_dict[row]
            elif data.loc[row, FeatureHeaders.sample_type.value] == 'Copolymer':
                smiles_list = [s.strip() for s in data.loc[row, FeatureHeaders.smiles.value]]
                conc_sum = data.loc[row, "Weight 1"] + data.loc[row, "Weight 2"]
                conc_list = [data.loc[row, "Weight 1"] + (1.0-conc_sum), data.loc[row, "Weight 2"]]
                try:
                    fp_dict[row] = self._fingerprint_copolymer(smiles_list=smiles_list, conc_list=conc_list)
                    # fp_dict[row] = fp.fingerprint_any_polymer(smiles_list=smiles_list, conc_list=conc_list)
                except Exception as e:
                    print(e)
                    print(smiles_list, conc_list)
                    exit()
            elif data.loc[row, FeatureHeaders.sample_type.value] == 'Blend':
                smiles_list = [s.strip() for s in data.loc[row, FeatureHeaders.smiles.value]]
                conc_sum = data.loc[row, "Weight 1"] + data.loc[row, "Weight 2"]
                conc_list = [data.loc[row, "Weight 1"] + (1.0-conc_sum), data.loc[row, "Weight 2"]]
                fp_dict[row] = self._fingerprint_blend(smiles_list=smiles_list, conc_list=conc_list)

        ela = time.time() - start
        print(f"fp total time: ", ela)
        
        fp_df = pd.DataFrame.from_dict(fp_dict, orient ='index').fillna(0)

        return fp_df

    def train_models(self):
        for model in self.models:
            model.set_scalers(self.scalers.copy())
            model.set_path(os.path.join(self.sim_dir, "models"))
            print(f"Training {model.name}")
            try:
                model.train_cv(self.train_df, self.train_df_fp)
            except Exception as e:
                model.logger.exception(e)
                print(f"{model.name} training failed, see log file for info")
                raise e
            print(f"Evaluating {model.name}")
            model.evaluate(self.test_df, self.test_df_fp)
    
    def train_models_parallel(self):
        #multi_inputs = [(self, m) for m in self.models]
        # Create a multiprocessing pool and parallelize the execution
        with multiprocessing.get_context('spawn').Pool(processes=len(self.models)) as pool:
            pool.map(self._train_single_model,self.models)

    def _train_single_model(self, model):
        model.set_scalers(self.scalers)
        model.set_path(os.path.join(self.sim_dir, "models"))
        print(f"Training {model.name}")
        try:
            model.train_cv(self.train_df, self.train_df_fp)
        except Exception as e:
            model.logger.exception(e)
            print(f"{model.name} training failed, see log file for info")
            exit()
        print(f"Evaluating {model.name}")
        model.evaluate(self.test_df, self.test_df_fp)

    def plot_data_split(self):
        sns.set_theme()

        sns.set(font_scale = 1.5, style = 'whitegrid')
        y_full = np.array(self.full_data[FeatureHeaders.visc.value])
        y_lim = (np.min(y_full) - 0.5, np.max(y_full) + 0.5)

        plotting_dir = os.path.join(TRAINING_DIR, self.name, DATASET_DIR)


        train_df = self.train_df.copy()
        train_df['Dataset'] = 'Train'
        test_df = self.test_df.copy()
        test_df['Dataset'] = 'Test'
        combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            # ax1.scatter(M_scaler.inverse_transform(M_test), y_scaler.inverse_transform(y_test), edgecolors= 'black')
        Mw_plot = sns.jointplot(data=combined_df, x=FeatureHeaders.mol_weight.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
        Mw_plot.ax_marg_y.remove()
        Mw_plot.ax_joint.set_xticks([3,5,7], [rf'$10^{i}$' for i in [3,5,7]], fontsize = 15)
        Mw_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize = 15)
        Mw_plot.set_axis_labels(r'$M_w$ (g/mol)', r'$\eta$ (Poise)')
        Mw_plot.ax_joint.set(ylim = y_lim)
        
        plt.savefig(os.path.join(plotting_dir, "Mw_plot.png"))

        for test_id in test_df['test_id'].unique():
            test_sample_unselected_df = self.test_samples_unselect.copy()[self.test_samples_unselect['test_id'] == test_id]
            test_sample_unselected_df['Dataset'] = 'Train_Samples'
            test_df = self.test_df.copy()[self.test_df['test_id'] == test_id]
            test_df['Dataset'] = 'Test'
            print("test_df: ", test_df)
            combined_df = pd.concat([test_sample_unselected_df, test_df]).reset_index(drop=True)
                # ax1.scatter(M_scaler.inverse_transform(M_test), y_scaler.inverse_transform(y_test), edgecolors= 'black')
            print(combined_df)
            Mw_plot = sns.jointplot(data=combined_df, x=FeatureHeaders.mol_weight.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
            Mw_plot.ax_marg_y.remove()
            Mw_plot.ax_joint.set_xticks([3,5,7], [rf'$10^{i}$' for i in [3,5,7]], fontsize = 15)
            Mw_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize = 15)
            Mw_plot.set_axis_labels(r'$M_w$ (g/mol)', r'$\eta$ (Poise)')
            Mw_plot.ax_joint.set(ylim = y_lim)
            
            plt.savefig(os.path.join(plotting_dir, f"Mw_plot_unselected_{test_id}.png"))


        #Zero Shear

        zshear_df = combined_df[combined_df[FeatureHeaders.shear_rate.value] == 0]
        # for i in zshear_df.index:
        #     zshear_df.loc[i, ['Shear Rate (1/s)']] = zshear_df.loc[i, ['Shear Rate (1/s)']][0]
        #     zshear_df.loc[i, ['Melt Viscosity (Poise)']] = zshear_df.loc[i, ['Melt Viscosity (Poise)']][0]

        zshear_plot = sns.scatterplot(data=zshear_df, x=FeatureHeaders.shear_rate.value, y=FeatureHeaders.visc.value, hue='Dataset')
        #zshear_plot = sns.scatterplot(data = zshear_df, x = 'Shear Rate (1/s)', y = 'Melt Viscosity (Poise)')
        #zshear_plot.ax_marg_y.remove()
        zshear_plot.set_xticks([0], ['0'], fontsize = 15)
        zshear_plot.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize =15)
        zshear_plot.set(xlabel = None, ylabel = None , ylim = y_lim)

        plt.savefig(os.path.join(plotting_dir, "zshear_plot.png"))

        #Shear

        shear_df = combined_df[combined_df[FeatureHeaders.shear_rate.value] > 0]
        shear_df[FeatureHeaders.shear_rate.value] = np.log10(shear_df[FeatureHeaders.shear_rate.value])
        # for i in shear_df.index:
        #     shear_df.loc[i, ['Shear Rate (1/s)']] = shear_df.loc[i, ['Shear Rate (1/s)']][0]
        #     shear_df.loc[i, ['Melt Viscosity (Poise)']] = shear_df.loc[i, ['Melt Viscosity (Poise)']][0]

        #shear_plot = sns.jointplot(data = shear_df, x = 'Shear Rate (1/s)', y = 'Melt Viscosity (Poise)', kind = 'scatter', height = 5)
        shear_plot = sns.jointplot(data=shear_df, x=FeatureHeaders.shear_rate.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
        
        shear_plot.ax_marg_y.remove()
        shear_plot.ax_joint.set_xticks(list(np.arange(-4,7,4)),  [r'$10^{-4}$'] + [rf'$10^{i}$' for i in list(np.arange(0,7,4))], fontsize = 15)
        shear_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,13,4))], fontsize =15)
        shear_plot.set_axis_labels(r'$\dot{\gamma}$ (1/s)', r'$\eta$ (Poise)')
        shear_plot.ax_joint.set(yticklabels=[], ylabel = None, ylim = y_lim)

        plt.savefig(os.path.join(plotting_dir, "shear_plot.png"))

        #Temperature

        # temp_df = pd.DataFrame({'Temp': Temp.reshape(-1,), 'Visc': yy.reshape(-1,)})
        # for i in temp_df.index:
        #     temp_df.loc[i, ['Temp']] = temp_df.loc[i, ['Temp']][0]
        #     temp_df.loc[i, ['Visc']] = temp_df.loc[i, ['Visc']][0]
        temp_plot = sns.jointplot(data=combined_df, x=FeatureHeaders.temp.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
        #temp_plot = sns.jointplot(data = temp_df, x = 'Temp', y = 'Visc', kind = 'scatter', height = 5)
        temp_plot.ax_marg_y.remove()
        temp_plot.set_axis_labels(r'Temperature (K)', r'$\eta$ (Poise)')
        temp_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize = 15)
        temp_plot.ax_joint.set(ylim = y_lim, ylabel = None)

        plt.savefig(os.path.join(plotting_dir, "temp_plot.png"))

        for test_id in test_df['test_id'].unique():
            test_sample_unselected_df = self.test_samples_unselect.copy()[self.test_samples_unselect['test_id'] == test_id]
            test_sample_unselected_df['Dataset'] = 'Train_Samples'
            test_df = self.test_df.copy()[self.test_df['test_id'] == test_id]
            test_df['Dataset'] = 'Test'
            print("test_df: ", test_df)
            combined_df = pd.concat([test_sample_unselected_df, test_df]).reset_index(drop=True)
                # ax1.scatter(M_scaler.inverse_transform(M_test), y_scaler.inverse_transform(y_test), edgecolors= 'black')
            print(combined_df)
            temp_plot = sns.jointplot(data=combined_df, x=FeatureHeaders.temp.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
            temp_plot.ax_marg_y.remove()
            temp_plot.ax_joint.set_xticks([0, 200, 400, 600], [rf'$10^{i}$' for i in [0, 200, 400, 600]], fontsize = 15)
            temp_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize = 15)
            temp_plot.set_axis_labels(r'$M_w$ (g/mol)', r'$\eta$ (Poise)')
            temp_plot.ax_joint.set(ylim = y_lim)
            
            plt.savefig(os.path.join(plotting_dir, f"Mw_plot_unselected_{test_id}.png"))

        #PDI

        # PDI_df = pd.DataFrame({'PDI': PDI.reshape(-1,), 'Visc': yy.reshape(-1,)})
        # for i in PDI_df.index:
        #     PDI_df.loc[i, ['PDI']] = PDI_df.loc[i, ['PDI']][0]
        #     PDI_df.loc[i, ['Visc']] = PDI_df.loc[i, ['Visc']][0]

        #PDI_plot = sns.jointplot(data = PDI_df, x = 'PDI', y = 'Visc', kind = 'scatter', height = 5)
        PDI_plot = sns.jointplot(data=combined_df, x=FeatureHeaders.PDI.value, y=FeatureHeaders.visc.value, hue='Dataset', kind='scatter', height=5)
        PDI_plot.set_axis_labels(r'PDI', r'$\eta$ (Poise)')
        PDI_plot.ax_joint.set_yticks(list(np.arange(0,13,4)), [rf'$10^{i}$' for i in list(np.arange(0,9,4))] + [r'$10^{12}$'], fontsize = 15)
        PDI_plot.ax_joint.set(ylim = y_lim, ylabel = None)

        plt.savefig(os.path.join(plotting_dir, "PDI_plot.png"))

    def save_data_split(self) -> None:
        os.makedirs(os.path.join(self.sim_dir, "dataset"), exist_ok=True)
        # save each df in the data split
        self.full_data_fp.to_pickle(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl"))
        self.train_df.to_pickle(os.path.join(self.sim_dir, "dataset", "train_df.pkl"))
        self.test_df.to_pickle(os.path.join(self.sim_dir, "dataset", "test_df.pkl"))
        self.train_df_fp.to_pickle(os.path.join(self.sim_dir, "dataset", "train_df_fp.pkl"))
        self.test_df_fp.to_pickle(os.path.join(self.sim_dir, "dataset", "test_df_fp.pkl"))

    def load_data_split(self) -> None:
        self.full_data_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl"))
        self.train_df = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "train_df.pkl"))
        self.test_df = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "test_df.pkl"))
        self.train_df_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "train_df_fp.pkl"))
        self.test_df_fp = pd.read_pickle(os.path.join(self.sim_dir, "dataset", "test_df_fp.pkl"))
    
    def _data_split_exists(self) -> bool:
        """
        Internal function to check if the data split is recorded. True if it is.
        """
        return os.path.exists(os.path.join(self.sim_dir, "dataset", "full_data_fp.pkl")) and \
            os.path.exists(os.path.join(self.sim_dir, "dataset", "train_df.pkl"))

if __name__ == '__main__':
    # Set multiprocessing to spawn
    multiprocessing.set_start_method('spawn', force=True)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global args
    args = parser.parse_args()
    file = args.config
    with open(file) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    # Initialize Models to test
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale_bounds=(1e-2, 1e2))
    # gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=20)

    # Initialize the GPflow regressor with the RBF kernel
    # kernel = gpflow.kernels.RBF()
    # gpr = GPflowRegressor(kernel=kernel)
    

    training_name = "training_mw_split_3"
    models = [#GPRModel(name = 'GPR', model_obj= HyperParam_GPR()),
    # PENNModel(name = training_name + '_ANN', model_obj=Visc_ANN,
    #                                         device = device, lr = 1e-3, epochs = 1000, batch_size = 8, 
    #                                         reduce_lr_factor = 0.5),
    # PENNModel(name = training_name + '_PENN_WLF_Hybrid', model_obj=Visc_PENN_WLF_Hybrid, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
                                                                            # reduce_lr_factor = 0.5, apply_gradnorm = False),
    PENNModel(name = training_name + '_PENN_WLF_critadj', model_obj=Visc_PENN_WLF, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
                                                                            reduce_lr_factor = 0.5, apply_gradnorm = False),
    # PENNModel(name = training_name +'_PENN_WLF_SP', model_obj=Visc_PENN_WLF_SP, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
    #                                                                         reduce_lr_factor = 0.5, apply_gradnorm = False),
    PENNModel(name = training_name +'_PENN_Arrhenius_critadj', model_obj=Visc_PENN_Arrhenius, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
                                                                            reduce_lr_factor = 0.5, apply_gradnorm = False),]
    # PENNModel(name = training_name +'_PENN_Arrhenius_SP', model_obj=Visc_PENN_Arrhenius_SP, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
    #                                                                         reduce_lr_factor = 0.5, apply_gradnorm = False),]
    #   PENNModel(name = training_name +'_PENN_PI_WLF_gradnorm', model_obj=Visc_PENN_PI_WLF, device = device, lr = 1e-4, epochs = 1000, batch_size = 8, 
    #                                                                         reduce_lr_factor = 0.5, apply_gradnorm = True)]
    #PENNModel(name = 'PENN_PI_Arrhenius_lr1e-5_b8', model_obj=Visc_PENN_PI_Arrhenius, device = device, lr = 1e-5, epochs = 200, batch_size = 8),
    #PENNModel(name = 'PENN_WLF_SP_lr1e-5_b8', model_obj=Visc_PENN_WLF_SP, device = device, lr = 1e-5, epochs = 200, batch_size = 8),]

    trainer = ModelTrainer(training_name,args, models, data_split_type='polyphysics')

    trainer.train_models()
