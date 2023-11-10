import pandas as pd
import numpy as np
from .utils.model_utils import load_models, predict_all_cv
from pgfingerprinting import fp
import joblib
from .Visc_PIMI import Visc_PIMI
import os

MODEL_PATH = os.path.dirname("./MODELS")


def get_model_and_scalers(date = '2023-04-16', data_type =  'full_data'):
    """
    The function returns various models and scalers loaded from saved files based on the input date and
    data type.
    
    :param date: The date of the model and scalers to be loaded, defaults to 2023-02-10 (optional)
    :param data_type: The data_type parameter is the data identifying training of the model
    :return: The function `get_model_and_scalers` returns a tuple containing the following objects:
    - `PENNs`: a list of PyTorch neural network models
    - `y_scaler`: a scaler object used to scale the viscosity
    - `M_scaler`: a scaler object used to scale the Mw
    - `S_scaler`: a scaler object used to scale the shear
    """
    PENNs, _ = load_models(date, data_type, NN_models= [])
    y_scaler, M_scaler, S_scaler, T_scaler, P_scaler, fp_scaler, S_torch_scaler = joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/y_scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/M_scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/S_scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/T_scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/P_scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/scaler.save'), joblib.load(f'{MODEL_PATH}/MODELS/{date}_{data_type}/scalers/S_torch_scaler.save')
    return PENNs, y_scaler, M_scaler, S_scaler, T_scaler, P_scaler, fp_scaler, S_torch_scaler



#TO DO: Add a check to ensure units of vars re correct (temp is given in Kelvin)
def predict(SMILES, Mw, T, shear, PDI = 2.06, return_raw = False):
    
    #load models and scalers
    PENNs, y_scaler, M_scaler, S_scaler, T_scaler, P_scaler, fp_scaler, S_torch_scaler = get_model_and_scalers()
    
    #get fingerprint, for features that the fp doesn't  have, add them as 0.
    fingp = fp.fingerprint_from_smiles(SMILES)
    fingp = pd.DataFrame(fingp, index = [0])
    for sc_col in fp_scaler.feature_names_in_:
        if sc_col not in fingp.columns:
            fingp[sc_col] = 0.
    
    #scale the FPs and physical variable
    fingp = fp_scaler.transform(fingp[fp_scaler.feature_names_in_])
    Mw = y_scaler.transform(np.log10(np.array(Mw )+ 0.00001).reshape(-1,1))
    shear = y_scaler.transform(np.log10(np.array(shear).reshape(-1,1)+ 0.00001))
    T = T_scaler.transform(np.array(T).reshape(-1,1))
    PDI = P_scaler.transform(np.ones_like(T)*PDI)
    print(np.log10(np.array(Mw)+ 0.00001).reshape(-1,1))
    
    #broadcast the fingerprint across all samples
    bcast = np.array(fingp).reshape(-1,)[np.newaxis, :] #* np.ones_like(np.array(Mw))
    all_fp = bcast* np.ones_like(np.array(Mw))


    preds, vars, _ = predict_all_cv(PENNs, [all_fp, Mw, shear, T, PDI], is_torch = True)
    if return_raw:
        return preds, vars
    else:
        #TO DO: unscale the the predicted values
        pass


    
    