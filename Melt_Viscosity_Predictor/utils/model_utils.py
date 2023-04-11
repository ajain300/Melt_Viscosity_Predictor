import torch
import numpy as np
import keras

class history_obj:
    def __init__(self, hist):
        self.history = hist

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
        X = [torch.tensor(x).to(models[0].device).float() for x in X_in]
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
                NN[i].append(keras.models.load_model(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/model_{n}'))
                history[i].append(history_obj(np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/hist_{n}.npy', allow_pickle=True).item()))
            except FileNotFoundError:
                print(f'Could not find ANN files for {date}_{data_type} {i}...')
                continue
        NN_cv = np.load(f'MODELS/{date}_{data_type}/{NN_models[i].__name__}/OME_CV.npy')
            
    
    
        for n in range(10):
            try:
                gpr.append(tf.saved_model.load(f'MODELS/{date}_{data_type}/GPR/model_{n}'))
            except FileNotFoundError:
                print(f'Could not find GPR files for {date}_{data_type} {n}...')
                continue

        gp_cv = np.load(f'MODELS/{date}_{data_type}/GPR/OME_CV.npy')

    for i in range(10):
        HypNet.append(torch.load(f'MODELS/{date}_{data_type}/PNN/model_{i}.pt'))
    HypNet_cv = np.load(f'MODELS/{date}_{data_type}/PNN/OME_CV.npy')

    return NN, history, gpr, gp_cv, NN_cv, HypNet, HypNet_cv


