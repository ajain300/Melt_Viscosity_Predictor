import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import keras
import pickle
import json

from Melt_Viscosity_Predictor.Visc_PIMI import Visc_PIMI

class history_obj:
    def __init__(self, hist):
        self.history = hist
class MVDataset_predict(Dataset):
  def __init__(self, FP, M, S, T, PDI):
    self.FP = FP
    self.M = M
    self.S = S
    self.T = T
    self.PDI = PDI

  def __len__(self):
    return len(self.PDI)
  
  def __getitem__(self,idx):
    return self.FP[idx], self.M[idx], self.S[idx], self.T[idx], self.PDI[idx]


def batch_predict(model, x):
    """
    This function is called when then requested prediction samples are too large (>10000). This will create batches of size 1000 to reduce memory usage.
    The function `batch_predict` takes a PyTorch model and input data, creates a data loader, performs a
    forward pass on the model for each batch of data, and returns the concatenated outputs.
    
    :param model: The PyTorch model that you want to use for prediction
    :param x: The input data for prediction. It could be a tuple, list, or any other data structure that
    contains the necessary information for the prediction. The specific format of x depends on the
    implementation of the MVDataset_predict class
    :return: The function `batch_predict` returns a PyTorch tensor `all_results` which contains the
    predictions made by the input `model` on the input `x`. The shape of `all_results` is `(num_samples,
    1)`, where `num_samples` is the total number of samples in the input `x`.
    """
    dataset = MVDataset_predict(*x)  # Replace YourDataset with your own dataset class
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    all_results = torch.empty((0, 1)).to(model.device)
    with torch.no_grad():
        for batch in dataloader:
            batch = [x_.to(model.device) for x_ in batch]
            outputs = model(*batch)  # Forward pass
            all_results = torch.cat((all_results, outputs), dim=0)  # Concatenate the results to the all_results tensor

    return all_results

def predict_all_cv(models, X_in, remove_model = None, is_torch = False, get_constants = False):
    """
    Gets mean and variance of predictions from all CV folds.
    """

    if not is_torch:
        # if remove_model:
        #     for r in remove_model: model_nums.pop(r)
        pred = []
        
        for m in models:
            #for layer in m.layers:
                #if isinstance(layer, tf.keras.layers.Dropout) and hasattr(layer, 'training'):
                 #   layer.training = True
            pred += [m(X_in, training = True)]
    else:
        X = [torch.tensor(x).float() for x in X_in]
        pred = []
        const_list = ['a1', 'a2', 'Mcr', 'kcr', 'c1', 'c2', 'Tr','tau', 'n', 'eta_0']
        constants = {k:[] for k in const_list}
        for m in models:
            for mod in m.modules():
                    if mod.__class__.__name__.startswith('Dropout'):
                        mod.train()
            if get_constants:
                y_pred, *const_out = m(*X, get_constants = True)
                y_pred = y_pred.cpu().detach().numpy()
                for k,i in zip(constants, range(len(const_out))):
                    constants[k] += [const_out[i].cpu().detach().numpy()]
            else:
                if X[0].shape[0] > 10000:
                    y_pred = batch_predict(m, X).cpu().detach().numpy()
                else:
                    X = [x_.to(m.device) for x_ in X]
                    y_pred = m(*X).cpu().detach().numpy()

            
            #print('FOLD PREDICTION', y_pred)
            if not np.any(y_pred < -3):
                pred += [y_pred]
    
    pred = np.array(pred)
    std = np.std(pred, axis = 0)
    means = np.nanmean(pred, axis = 0)
    if get_constants:
        for k in constants:
            constants[k] = np.array(constants[k])
            constants[k] = np.nanmean(constants[k], axis = 0)
    try:
        if not get_constants:
            return [m[0] for m in means], [s[0] for s in std], pred
        else:
            return constants
    except:
        print(means)
        print(type(means))
        print('Error in prediction')

def load_models(date, data_type, NN_models: list):
    NN = [[],[]]
    HypNet = []
    gpr = []
    history = [[],[]]
    for i in range(len(NN_models)):
        
        for n in range(10):
            try:
                NN[i].append(keras.models.load_model(f'../MODELS/{date}_{data_type}/{NN_models[i].__name__}/model_{n}'))
                history[i].append(history_obj(np.load(f'../MODELS/{date}_{data_type}/{NN_models[i].__name__}/hist_{n}.npy', allow_pickle=True).item()))
            except FileNotFoundError:
                print(f'Could not find ANN files for {date}_{data_type} {i}...')
                continue
        NN_cv = np.load(f'../MODELS/{date}_{data_type}/{NN_models[i].__name__}/OME_CV.npy')
            
    
    
        for n in range(10):
            try:
                gpr.append(tf.saved_model.load(f'MODELS/{date}_{data_type}/GPR/model_{n}'))
            except FileNotFoundError:
                print(f'Could not find GPR files for {date}_{data_type} {n}...')
                continue

        gp_cv = np.load(f'../MODELS/{date}_{data_type}/GPR/OME_CV.npy')

    #with open(f'../MODELS/{date}_{data_type}/PNN/n_features.pickle', 'wb') as file:
    n_features = 219 #pickle.load(file)
    for i in range(10):
        with open(f'./MODELS/{date}_{data_type}/PNN/hyperparam_{i}.json') as file:
            config = json.load(file)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Visc_PIMI(n_features, config, device)
        model.load_state_dict(torch.load(f'./MODELS/{date}_{data_type}/PNN/model_{i}.pt'))
        model.eval()
        HypNet.append(model)
    HypNet_cv = np.load(f'./MODELS/{date}_{data_type}/PNN/OME_CV.npy')

    if len(NN_models) == 0:
        return HypNet, HypNet_cv
    else:
        return NN, history, gpr, gp_cv, NN_cv, HypNet, HypNet_cv


