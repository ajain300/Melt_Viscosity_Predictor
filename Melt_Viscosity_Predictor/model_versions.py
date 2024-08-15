import torch
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping as EarlyStoppingKeras
from utils.train_utils import FeatureHeaders, hyperparam_opt_ray
from utils.metrics import OME
from utils.model_utils import batch_predict
from data_tools.curve_fitting import fit_WLF, WLF_obj
from torch.utils.data.dataloader import DataLoader
from utils.train_torch import MVDataset, run_training
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from Visc_PENN import *
from ViscNN import ViscNN_concat_HP
import wandb
from wandb.integration.keras import WandbMetricsLogger
from joblib import dump, load
import pandas as pd
import os
import logging
import json
import pickle
import contextlib
import io

from typing import Dict

MODEL_FOLDER = "model_fold"

# BaseModel class generalizable for training all model variations
class BaseModel:
    def __init__(self, name, model_obj, n_splits=10, random_state = 0):
        self.name = name
        self.model = model_obj
        self.models = [None] * n_splits  # Placeholder for CV model instances
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.hyperparams = [None] * n_splits
        self.train_history = [None] * n_splits
        self.cv_fold_validation_OME = [None] * n_splits

    def set_scalers(self, scalers : Dict[str, MinMaxScaler]):
        self.scalers = scalers

    def set_path(self, model_path : str):
        self.model_path = os.path.join(model_path, self.name)
        os.makedirs(self.model_path, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        def make_log(log_filename):
            # Create a logger for MyClassA
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Add a file handler to the logger
            handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            return logger
        
        self.log_file = os.path.join(self.model_path, f"{self.name}.log")
        self.logger = make_log(self.log_file)

    def transform_data(self, df : pd.DataFrame, df_fp: pd.DataFrame, train_data = True):
        self.logger.info("Transforming Data")
        fp = np.array(self.scalers[FeatureHeaders.fp.value].transform(df_fp.copy()))
        
        logMw = np.array(df[FeatureHeaders.mol_weight.value], dtype = float).reshape((-1,1))
        M = self.scalers[FeatureHeaders.mol_weight.value].transform(logMw)
        
        shear = np.array(df[FeatureHeaders.shear_rate.value], dtype = float).reshape((-1,1))
        S = self.scalers[FeatureHeaders.shear_rate.value].transform(np.log10(shear + 0.00001))
        
        Temp = np.array(df[FeatureHeaders.temp.value]).reshape((-1,1))
        T = self.scalers[FeatureHeaders.temp.value].transform(Temp)

        PDI = np.array(df[FeatureHeaders.PDI.value]).reshape((-1,1))
        P = self.scalers[FeatureHeaders.PDI.value].transform(np.log(PDI))

        if train_data:
            yy = np.array(df.loc[:,FeatureHeaders.visc.value]).reshape((-1,1))
            yy = self.scalers[FeatureHeaders.visc.value].transform(yy)
            return fp, yy, M, S, T, P

        return fp, M, S, T, P
        
    def train_cv(self, train_df : pd.DataFrame, train_df_fp : pd.DataFrame):
        fold = 0
        fp, yy, M, S, T, P = self.transform_data(train_df, train_df_fp.copy())

        for train_index, test_index in self.cv.split(fp):
            # create folder for this split
            os.makedirs(os.path.join(self.model_path, f"fold_{fold}"), exist_ok = True)
            
            fold_data_path = os.path.join(self.model_path, f"fold_{fold}", "data_split.pkl")
            if not os.path.exists(fold_data_path):
                fp_train, fp_val = fp[train_index], fp[test_index]
                y_train, y_val = yy[train_index], yy[test_index]
                M_train, M_val = M[train_index], M[test_index]
                S_train, S_val = S[train_index], S[test_index]
                T_train, T_val = T[train_index], T[test_index]
                P_train, P_val = P[train_index], P[test_index]
                self.fp_size = fp_train.shape[1]

                fit_X = [fp_train, M_train, S_train, T_train, P_train]
                eval_X = [fp_val, M_val, S_val, T_val, P_val]

                # save data split
                with open(fold_data_path, 'wb') as f:
                    pickle.dump({"fit_X": fit_X,
                                "eval_X": eval_X,
                                "fit_Y": y_train,
                                "eval_Y": y_val,
                                "fp_size": self.fp_size}, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # load data split
                with open(fold_data_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    fit_X = loaded_data["fit_X"]
                    eval_X = loaded_data["eval_X"]
                    y_train = loaded_data["fit_Y"]
                    y_val = loaded_data["eval_Y"]
                    self.fp_size = loaded_data["fp_size"]
                
            self.train_fold(fold, fit_X, eval_X, y_train, y_val)
            fold += 1
        
        # evaluate training points training parity
        self.logger.info("Eval train points")
        self.evaluate(train_df, train_df_fp, eval_type = 'train')

        if self.run:
            print("FINISHING WANDB RUN")
            self.run.finish()

    def train_fold(self, model_fold, fit_X, eval_X, fit_y, eval_y):
        raise NotImplementedError("Method 'train' must be implemented in subclasses")

    def predict_fold(self, model_fold, X_test, do_dropout = False):
        raise NotImplementedError("Method 'predict' must be implemented in subclasses")
    
    def predict_cv(self, X):
        raise NotImplementedError("Method 'predict_cv' must be implemented in subclasses")

    def inference(self, df, df_fp):
        fp, M, S, T, P = self.transform_data(df, df_fp, train_data=False)
        pred_mean, pred_std = self.predict_cv([fp, M, S, T, P], S = S, M= M)
        pred_mean_scaled = self.scalers[FeatureHeaders.visc.value].inverse_transform(pred_mean)
        pred_std_scaled = pred_mean_scaled - self.scalers[FeatureHeaders.visc.value].inverse_transform(pred_mean-pred_std)
        return pred_mean_scaled, pred_std_scaled
    
    def evaluate(self, test_df : pd.DataFrame, test_df_fp : pd.DataFrame, **kwargs):
        pred_mean, pred_std = self.inference(test_df, test_df_fp)
        eval_type = kwargs.get("eval_type", "test")
        self.plot_parity(test_df[FeatureHeaders.visc.value],pred_mean, pred_std, plot_name = eval_type)

        test_df = test_df.append(pd.DataFrame(pred_mean, columns = [FeatureHeaders.pred_mean.value]), ignore_index = True)
        test_df = test_df.append(pd.DataFrame(pred_std, columns = [FeatureHeaders.pred_std.value]), ignore_index = True)

        eval_save_path = os.path.join(self.model_path, f"{eval_type}_results.pkl")
        test_df.to_pickle(eval_save_path)
   
    def plot_parity(self, y_test, test_pred, test_var, plot_name = 'test'):
        plt.figure(figsize = (5.5,5.5))

        plt.errorbar(y_test , list(test_pred.reshape(-1,)), yerr= list(np.array(test_var).reshape(-1,)), fmt =  'o', label = f'Test: {y_test.shape[0]} datapoints, ' 
        + r'$R^2$ = ' + "{:1.3f}, ".format(r2_score(y_test, test_pred)) + "OME = {:1.4f}".format(OME(y_test, test_pred)))

        axis_min = min(y_test.min(), (test_pred - test_var).min())
        axis_max = min(y_test.max(), (test_pred + test_var).max())

        plt.plot(np.linspace(axis_min, axis_max, num = 2),np.linspace(axis_min, axis_max, num = 2),'k-', zorder = 10)
        plt.ylabel(r'$Log$ Viscosity (Poise) ML Predicted')
        plt.xlabel(r'$Log$ Viscosity (Poise) Experimental Truth')
        plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
        plt.title(f'{self.name} Parity Plot')
        plt.xlim(-2, 12.5)
        plt.ylim(-2, 12.5)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(os.path.join(self.model_path, f"{plot_name}_parity.png"))

    def load_model(self):
        raise NotImplementedError("Need to implement load_model for specific model type.")

class GPRModel(BaseModel):
    # Overwrite the original train_cv code
    # Hyperparameter space for Bayesian optimization
    search_spaces = {
        "alpha": Real(1e-10, 1e1, prior='log-uniform'),  # Noise level
        "length_scale": Real(1e-2, 1e2, prior='log-uniform'),  # RBF length scale
        "constant_value": Real(1e-2, 1e2, prior='log-uniform')  # Constant value for kernel
    }

    def train_cv(self, train_df : pd.DataFrame, train_df_fp : pd.DataFrame):
        model_dir = os.path.join(self.model_path, "gpr_model.pkl")
        
        if not os.path.exists(model_dir):
            fp, yy, M, S, T, P = self.transform_data(train_df, train_df_fp)
            X_train_ = np.concatenate((fp, M, S, T, P), axis = 1)
            # Grid search across CV folds
            verbose_output = io.StringIO()
            with contextlib.redirect_stdout(verbose_output):
                self.bayes_search = BayesSearchCV(self.model, self.search_spaces, n_iter=50, cv=self.cv, n_jobs=-1, verbose =1)
            verbose_content = verbose_output.getvalue()
            self.logger.info(verbose_content)
            self.bayes_search.fit(X_train_, yy)
            dump(self.bayes_search, model_dir)
        else:
            self.logger.info("Loaded GPR model.")
            self.bayes_search = load(model_dir)
        
        # evaluate training points training parity
        self.logger.info("Eval train points")
        self.evaluate(train_df, train_df_fp, eval_type = 'train')

    def predict_cv(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = np.concatenate(X, axis = 1)
        y_mean, y_pred_std = self.bayes_search.best_estimator_.predict(X, return_std = True)

        return y_mean.reshape(-1,1), y_pred_std.reshape(-1,1)

    def load_model(self):
        assert self.model_path, "Need to set model_path before loading model."
        model_dir = os.path.join(self.model_path, "gpr_model.pkl")
        self.bayes_search = load(model_dir)

class ANNModel(BaseModel):
    def __init__(self, name, model_obj, n_splits=10, random_state = 0, epochs = 500, **kwargs):
        super().__init__(name, model_obj, n_splits, random_state)
        self.epochs = epochs
        self.batch_size = kwargs.get("batch_size", 32)
        self.lr = kwargs.get("lr", 1e-3)    
        wandb.init(project="Melt_Visc_PENN", name = name)   
        config = wandb.config
        config.learning_rate = self.lr
        config.epochs = epochs
        config.batch_size = self.batch_size

    def setup_logging(self):
        """
        Sets up logging and initialized wandb run.
        """
        def make_log(log_filename):
            # Create a logger for MyClassA
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Add a file handler to the logger
            handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            return logger
        
        self.log_file = os.path.join(self.model_path, f"{self.name}.log")
        self.logger = make_log(self.log_file)
        self.run = None

    def train_fold(self, model_fold, fit_X, eval_X, fit_y, eval_y):
        model_fold_path = os.path.join(self.model_path, f"fold_{model_fold}", "model.h5")
        model_hist_path = os.path.join(self.model_path, f"fold_{model_fold}", "history.pkl")
        if not os.path.exists(model_fold_path):
            self.logger.info(f"Training {self.name} fold {model_fold}")
            fold_hyper_params = self.hyperparam_opt(ViscNN_concat_HP, fit_X, eval_X, fit_y, eval_y, model_fold, self.name)
            self.hyperparams[model_fold] = fold_hyper_params.values
            self.logger.info(f'hyperparams',fold_hyper_params.values)
            
            # Define early stopping
            early_stopping = EarlyStoppingKeras(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            )
            
            self.models[model_fold] = self.model(self.fp_size, fold_hyper_params)
            self.train_history[model_fold] = self.models[model_fold].fit(fit_X, fit_y, epochs=self.epochs, 
                                                                        batch_size=self.batch_size, 
                                                                        validation_data = (eval_X, eval_y), verbose=0, 
                                                                        callbacks = [early_stopping, WandbMetricsLogger()])
            self.models[model_fold].save(model_fold_path)
            # plot training test fold
            eval_y_pred = self.models[model_fold](eval_X, training = True)
            plt.figure(figsize = (5.5,5.5))
            plt.scatter(eval_y, eval_y_pred)
            plt.plot(np.linspace(-1, 1, num = 2),np.linspace(-1, 1, num = 2),'k-', zorder = 10)
            plt.ylabel(r'Scaled $Log$ Viscosity (Poise) ML Predicted')
            plt.xlabel(r'Scaled $Log$ Viscosity (Poise) Experimental Truth')
            plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
            plt.gca().set_aspect('equal', adjustable='box')

            plt.savefig(os.path.join(self.model_path, f"fold_{model_fold}",f"test_fold{model_fold}_parity.png"))        

            dump(self.train_history[model_fold], model_hist_path)
        else:
            self.models[model_fold] = load_model(model_fold_path)
            self.train_history[model_fold] = load(model_hist_path)

    def eval_fold(self, model_fold, eval_X, eval_y):
        self.cv_fold_validation_OME = OME(eval_y, self.predict_fold(model_fold, eval_X))

    def load_model(self):
        assert self.model_path, "Need to set model_path before loading model."
        for fold, model in enumerate(self.models):
            model_fold_path = os.path.join(self.model_path, f"fold_{fold}", "model.h5")
            self.models[fold] = load_model(model_fold_path)
        
    def predict_fold(self, model_fold, X_test, do_dropout = False):
        pred = self.models[model_fold](X_test, training = do_dropout)
        return pred
    
    def predict_cv(self, X) -> tuple[np.ndarray, np.ndarray]:
        pred = []
        
        for m in self.models[1:]:
            for passes in range(5):
                pred += [m(X, training = True)]
        
        pred = np.array(pred)
        print("pred_cv out", pred.shape)
        print(pred[: , :5])

        std = np.std(pred, axis = 0)
        means = np.nanmean(pred, axis = 0)

        print(means)

        #return [m[0] for m in means], [s[0] for s in std], pred
        return np.array([m[0] for m in means]).reshape(-1,1), np.array([s[0] for s in std]).reshape(-1,1)
 
    def hyperparam_opt(self, hypermodel, fit_in, eval_in, y_train, y_val, iter, train_type):
        hp_dir = os.path.join(self.model_path, "hyperparam_search")
        os.makedirs(hp_dir, exist_ok=True)
        model = hypermodel(fit_in[0].shape[1])

        tuner = kt.Hyperband(model, objective='val_loss',
                            max_epochs=30, hyperband_iterations = 1,
                            factor=3, project_name = f'{train_type}_{iter}', directory=hp_dir)

        tuner.search(fit_in, y_train,
                    validation_data= (eval_in, y_val),
                    epochs=30,
                    batch_size = 20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
        
        return tuner.get_best_hyperparameters(1)[0]

class PENNModel(BaseModel):
    def __init__(self, name, model_obj : Visc_PENN_Base, device,n_splits=10, random_state = 0, epochs = 500, apply_gradnorm = False, training = True, **kwargs):
        super().__init__(name, model_obj, n_splits, random_state)
        
        self.epochs = epochs
        self.device = device
        self.training = training
        self.apply_gradnorm = apply_gradnorm
        self.reduce_lr_factor = kwargs.get("reduce_lr_factor", 0.1)
        self.batch_size = kwargs.get("batch_size", 32)
        self.penn_type = 'WLF' if model_obj in [Visc_PENN_WLF, Visc_PENN_PI_WLF, Visc_PENN_WLF_SP] else 'Arr'
        self.lr = kwargs.get("lr", 1e-4)        
    
    def setup_prediction_data(self):
        self.prediction_data = {}

    def setup_logging(self):
        """
        Sets up logging and initialized wandb run.
        """
        def make_log(log_filename):
            # Create a logger for MyClassA
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

            # Add a file handler to the logger
            handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            return logger
        
        self.log_file = os.path.join(self.model_path, f"{self.name}.log")
        self.logger = make_log(self.log_file)
        
        if self.training:
            self.run = wandb.init(project="Melt_Visc_PENN", name = self.name,
            config = {
                "learning_rate": self.lr,
                "batch_size": self.batch_size
            })

    def train_fold(self, model_fold, fit_X, eval_X, fit_y, eval_y):
        self.logger.info(f"Training {self.name} fold {model_fold}")
        
        # Setup dataloaders
        fit_ten = [torch.tensor(a, requires_grad=True).to(self.device).float() for a in fit_X]
        val_ten = [torch.tensor(a).to(self.device).float() for a in eval_X]
        tr_load = DataLoader(MVDataset(*fit_ten, torch.tensor(fit_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
        val_load = DataLoader(MVDataset(*val_ten, torch.tensor(eval_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
        self.logger.info(f"Set up dataloaders")
        # If a configuration alreayd exists for this fold then use that else run hp tuning
        config_path = os.path.join(self.model_path, f"fold_{model_fold}", "best_config.json")
        if not os.path.exists(config_path):
            # Hyperparam tuning
            best_config = hyperparam_opt_ray(self.fp_size, tr_load, val_load, self.device, self.model, self.logger, lr = self.lr, name = self.name + f"_fold{model_fold}")
            self.logger.info('Best Configuration from RayTune ', best_config)

            # save config
            with open(config_path, 'w') as f:
                json.dump(best_config, f, indent=4)
        else:
            with open(config_path, 'r') as f:
                best_config = json.load(f)

        self.logger.info("Running training.")
        # training
        checkpoint_folder = os.path.join(self.model_path, f"fold_{model_fold}")
        model = run_training(best_config, n_fp = self.fp_size, 
                                                    train_loader = tr_load, 
                                                    test_loader = val_load, 
                                                    device = self.device, 
                                                    EPOCHS = self.epochs, 
                                                    model_arch = self.model, 
                                                    ckpt_folder = checkpoint_folder, 
                                                    logger = self.logger,
                                                    apply_gradnorm = self.apply_gradnorm,
                                                    lr = self.lr,
                                                    run = self.run,
                                                    fold = model_fold,
                                                    reduce_lr_factor = self.reduce_lr_factor)
        self.models[model_fold] = model
        self.hyperparams = best_config

        # plot training test fold
        eval_y_pred = model(*val_ten).detach().cpu().numpy()
        plt.figure(figsize = (5.5,5.5))
        plt.scatter(eval_y, eval_y_pred)
        plt.plot(np.linspace(-1, 1, num = 2),np.linspace(-1, 1, num = 2),'k-', zorder = 10)
        plt.ylabel(r'Scaled $Log$ Viscosity (Poise) ML Predicted')
        plt.xlabel(r'Scaled $Log$ Viscosity (Poise) Experimental Truth')
        plt.legend(loc = 'upper left', frameon = False, prop={"size":8})
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(os.path.join(self.model_path, f"fold_{model_fold}",f"test_fold{model_fold}_parity.png"))        

    def set_scalers(self, scalers) -> None:
        self.scalers = scalers
        self.scalers[FeatureHeaders.mol_weight.value] = scalers[FeatureHeaders.visc.value]
        self.scalers[FeatureHeaders.shear_rate.value] = scalers[FeatureHeaders.visc.value]
        if self.penn_type == 'Arr':
            self.scalers[FeatureHeaders.temp.value] = scalers[FeatureHeaders.inv_temp.value]

    def transform_data(self, df : pd.DataFrame, df_fp: pd.DataFrame, train_data = True):
        
        fp = np.array(self.scalers[FeatureHeaders.fp.value].transform(df_fp.copy()))
        
        logMw = np.array(df[FeatureHeaders.mol_weight.value], dtype = float).reshape((-1,1))
        M = self.scalers[FeatureHeaders.mol_weight.value].transform(logMw)
        
        shear = np.array(df[FeatureHeaders.shear_rate.value], dtype = float).reshape((-1,1))
        S = self.scalers[FeatureHeaders.shear_rate.value].transform(np.log10(shear + 0.00001))
        
        if self.penn_type == 'WLF':
            Temp = np.array(df[FeatureHeaders.temp.value]).reshape((-1,1))
            T = self.scalers[FeatureHeaders.temp.value].transform(Temp)
        else: 
            Temp = np.array(df[FeatureHeaders.temp.value]).reshape((-1,1))
            T = self.scalers[FeatureHeaders.temp.value].transform(1/Temp)

        PDI = np.array(df[FeatureHeaders.PDI.value]).reshape((-1,1))
        P = self.scalers[FeatureHeaders.PDI.value].transform(np.log(PDI))

        if train_data:
            yy = np.array(df.loc[:,FeatureHeaders.visc.value]).reshape((-1,1))
            yy = self.scalers[FeatureHeaders.visc.value].transform(yy)
            return fp, yy, M, S, T, P

        return fp, M, S, T, P

    def predict_fold(self, model_fold, X_test):
        return
    
    def predict_cv(self, X, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Upon prediction, we do, 5 passes for each fold, averaging and taking the std of all 10 folds
        """
        X = [torch.tensor(x).float() for x in X]
        pred = []
        M = kwargs.get('M', np.array([]))

        fig = plt.figure()
        for model_num, m in enumerate(self.models):
            if m:
                for mod in m.modules():
                    if mod.__class__.__name__.startswith('Dropout'):
                        mod.train()
                for passes in range(5):
                    if X[0].shape[0] > 10000:
                        y_pred = batch_predict(m, X).cpu().detach().numpy()
                    else:
                        X = [x_.to(m.device) for x_ in X]
                        y_pred = m(*X).cpu().detach().numpy()
                    pred += [y_pred]
                    if M.shape[0] != 0: 
                        plt.plot(M.reshape(-1,), y_pred.reshape(-1,))
        
        plt.savefig(os.path.join(self.model_path, "cv_pred.png"))
        plt.close()
                        
        pred = np.array(pred)
        std = np.std(pred, axis = 0)
        means = np.nanmean(pred, axis = 0)

        return np.array([m[0] for m in means]).reshape(-1,1), np.array([s[0] for s in std]).reshape(-1,1)

    def get_constants(self, X):
        
        if self.penn_type == 'WLF':
            const_list = [Visc_Constants.a1.value,
                    Visc_Constants.a2.value, 
                    Visc_Constants.Mcr.value, 
                    Visc_Constants.k1.value, 
                    Visc_Constants.c1.value, 
                    Visc_Constants.c2.value, 
                    Visc_Constants.Tr.value, 
                    Visc_Constants.Scr.value, 
                    Visc_Constants.n.value, 
                    Visc_Constants.beta_M.value,
                    Visc_Constants.beta_shear.value,]
        else:
            const_list = [Visc_Constants.a1.value,
                    Visc_Constants.a2.value, 
                    Visc_Constants.Mcr.value, 
                    Visc_Constants.k1.value, 
                    Visc_Constants.lnA.value, 
                    Visc_Constants.EaR.value, 
                    Visc_Constants.Scr.value, 
                    Visc_Constants.n.value, 
                    Visc_Constants.beta_M.value,
                    Visc_Constants.beta_shear.value,]
        
        constants = {k:[] for k in const_list}

        X = [torch.tensor(x).float() for x in X]

        for m in self.models:
            if m:
                X = [x.to(m.device) for x in X]
                const_out = m(*X, get_constants = True)
                for k,i in const_out.items():
                    constants[k] += [const_out[k].cpu().detach().numpy()]
        
        for k in constants:
            constants[k] = np.array(constants[k])
            constants[k] = np.nanmean(constants[k], axis = 0).reshape(-1,1)
        
        return constants
    
    def predict_constants(self, df : pd.DataFrame, df_fp : pd.DataFrame):
        fp, M, S, T, P = self.transform_data(df, df_fp, train_data=False)
        constants = self.get_constants([fp, M, S, T, P])
        constants = self.transform_constants(constants)
        return constants

    def transform_constants(self, constants): 
        """
        Scale back constants to original scaling.
        """
        constants[Visc_Constants.Mcr.value] = self.scalers[FeatureHeaders.mol_weight.value].inverse_transform(constants[Visc_Constants.Mcr.value])
        constants[Visc_Constants.Scr.value] = self.scalers[FeatureHeaders.shear_rate.value].inverse_transform(constants[Visc_Constants.Scr.value])
        
        # Transform WLF constants
        if Visc_Constants.c1.value in constants.keys():
            t_sc_range = np.linspace(-1, 1)
            
            for i, (c1, c2, tr) in enumerate(zip(constants[Visc_Constants.c1.value], constants[Visc_Constants.c2.value], constants[Visc_Constants.Tr.value])):
                scaled_wlf = WLF_obj(t_sc_range, tr, c1, c2, 0)
                unsc_wlf = self.scalers[FeatureHeaders.visc.value].inverse_transform(scaled_wlf.reshape(-1,1))
                unsc_temp = self.scalers[FeatureHeaders.temp.value].inverse_transform(t_sc_range.reshape(-1,1))
                unsc_const, _ = fit_WLF(unsc_temp.reshape(-1,), unsc_wlf.reshape(-1,))
                constants[Visc_Constants.c1.value][i] = unsc_const[Visc_Constants.c1.value]
                constants[Visc_Constants.c2.value][i] = unsc_const[Visc_Constants.c2.value]
                constants[Visc_Constants.Tr.value][i] = unsc_const[Visc_Constants.Tr.value]

        return constants
        
    def load_model(self):
        assert self.model_path, "Need to set model_path before loading model."

        for fold, model in enumerate(self.models):
            try:
                # Get the fp size from fold path
                fold_data_path = os.path.join(self.model_path, f"fold_{fold}", "data_split.pkl")
                with open(fold_data_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                self.fp_size = loaded_data["fp_size"]
                
                config_path = os.path.join(self.model_path, f"fold_{fold}", "best_config.json")
                ckpt_path = os.path.join(self.model_path, f"fold_{fold}", "checkpoint.pt")
                
                with open(config_path, 'r') as f:
                    best_config = json.load(f)
                
                model = self.model(self.fp_size, best_config, self.device).to(self.device)
                model.load_trained_model(ckpt_path)
                self.models[fold] = model
            except Exception as e:
                print("Couldn't load model ", fold)
                print(e)

    def compute_loss_for_single_batch(self, net : Visc_PENN_Base):
        fold_data_path = os.path.join(self.model_path, f"fold_{0}", "data_split.pkl")
        with open(fold_data_path, 'rb') as f:
            loaded_data = pickle.load(f)
            fit_X = loaded_data["fit_X"]
            eval_X = loaded_data["eval_X"]
            fit_y = loaded_data["fit_Y"]
            eval_y = loaded_data["eval_Y"]
            self.fp_size = loaded_data["fp_size"]

        fit_ten = [torch.tensor(a, requires_grad=True).to(self.device).float() for a in fit_X]
        val_ten = [torch.tensor(a).to(self.device).float() for a in eval_X]
        tr_load = DataLoader(MVDataset(*fit_ten, torch.tensor(fit_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
        val_load = DataLoader(MVDataset(*val_ten, torch.tensor(eval_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
       
        val_loss, avg_val_loss = net.evaluate(tr_load, torch.nn.MSELoss())
        return avg_val_loss.cpu().detach().numpy()
    
    # def get_param_around_min(self):
    #     fold_data_path = os.path.join(self.model_path, f"fold_{0}", "data_split.pkl")
    #     with open(fold_data_path, 'rb') as f:
    #         loaded_data = pickle.load(f)
    #         fit_X = loaded_data["fit_X"]
    #         eval_X = loaded_data["eval_X"]
    #         fit_y = loaded_data["fit_Y"]
    #         eval_y = loaded_data["eval_Y"]
    #         self.fp_size = loaded_data["fp_size"]
        
    #     # Setup dataloaders
    #     fit_ten = [torch.tensor(a, requires_grad=True).to(self.device).float() for a in fit_X]
    #     val_ten = [torch.tensor(a).to(self.device).float() for a in eval_X]
    #     tr_load = DataLoader(MVDataset(*fit_ten, torch.tensor(fit_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
    #     val_load = DataLoader(MVDataset(*val_ten, torch.tensor(eval_y).to(self.device).float()), batch_size = self.batch_size, shuffle = True)
    #    # If a configuration alreayd exists for this fold then use that else run hp tuning
    #     config_path = os.path.join(self.model_path, f"fold_{model_fold}", "best_config.json")

    #     with open(config_path, 'r') as f:
    #         best_config = json.load(f)

    #     # training
    #     checkpoint_folder = os.path.join(self.model_path, f"fold_{model_fold}")
    #     model = run_training(best_config, n_fp = self.fp_size, 
    #                                                 train_loader = tr_load, 
    #                                                 test_loader = val_load, 
    #                                                 device = self.device, 
    #                                                 EPOCHS = self.epochs, 
    #                                                 model_arch = self.model, 
    #                                                 ckpt_folder = checkpoint_folder, 
    #                                                 logger = self.logger,
    #                                                 apply_gradnorm = self.apply_gradnorm,
    #                                                 lr = self.lr,
    #                                                 run = self.run,
    #                                                 fold = 0,
    #                                                 reduce_lr_factor = self.reduce_lr_factor,
    #                                                 save_params = True)



        


class HyperParam_GPR(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-10, length_scale=1.0, constant_value=1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.constant_value = constant_value

    def fit(self, X, y):
        # Define the kernel with the current hyperparameters
        kernel = C(self.constant_value, (1e-3, 1e3)) * RBF(self.length_scale, (1e-2, 1e2))
        self.gpr_ = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, n_restarts_optimizer=20)
        self.gpr_.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.gpr_.predict(X, return_std=return_std)

    def score(self, X, y):
        return self.gpr_.score(X, y)