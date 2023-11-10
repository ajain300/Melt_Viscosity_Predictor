import pytest
import pandas as pd
from  ..utils.eval_utils import *
import os

FOLDER_DIR = os.path.dirname(__file__)
TEST_DF = os.path.join(os.path.dirname(__file__), "sample_data.xlsx")

def test_Mw_extrap():
    sample_df = pd.read_excel(TEST_DF)
    sample_df = sample_df[['Polymer', 'Mw', 'Shear_Rate', 'Temperature', 'PDI','Melt_Viscosity'] + [c for c in list(sample_df.columns) if 'fp' in str(c)]]
    mw_df, mw_samps = get_Mw_samples(sample_df)
    assert len(mw_df) == 11
    assert mw_samps == [1000]
    mw_samps_answer = pd.read_pickle(F"{FOLDER_DIR}/answers/get_Mw_Samples.pkl")
    assert mw_df.equals(mw_samps_answer)
    
    
    
    
         


