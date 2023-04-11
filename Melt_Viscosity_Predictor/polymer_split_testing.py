from sklearn.model_selection import KFold, StratifiedKFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
from utils.metrics import MSE, OME
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras_tuner as kt
import json
from os import sys
sys.path.append('../')
from ViscNN import ViscNN_concat_HP
from utils.model_utils import predict_all_cv
import time
from math import floor
import yaml
import train_models
import re


def main():
    test_sizes = list(1 - np.linspace(0.1, 0.9, 9))
    print(test_sizes)
    train_error = []
    test_error = []
    poly_ind = 0
    polymers = ['poly(methyl methacrylate)']
    #polymers = ['poly[(butane-1,4-diol)-alt-(isophthalic acid)]', 'poly(oxybutane-1,4-diyloxy{5-[(sodiooxy)sulfonyl]isophthaloyl})', "poly[(2,2'-oxydiethanol)-alt-(adipic acid)"]
                
 #'polypent1ene',
 #'polyethenePoly1octene',
 #'LinearHDPE',
 #'polyethenePolybutene',
 
 #'polybisphenolAcoepichlorohydrin',]
#  'polyhexane16diaminealtadipicacid',
#  'polybutane14diolaltdiethylsuccinate',
#  'poly22334455octafluorohexane16diolaltglutaryldichloride',
#  'poly2chloromethyloxirane',
#  'polydodecano12lactam',
#  'polyoxydifluoromethylenepolyoxy1122tetrafluoroethylene',
#  'polypropane13diolaltdecanedioicacid',
#  'poly223344hexafluoropentane15diolaltadipicacid']
    id = 0
    for poly in polymers: #for each poly ind
        #change the config file to include the index

            with open('./config.yaml') as f:
                config = yaml.safe_load(f)
                f.close()
            
            poly = re.sub('[\W_]+', '', poly)
            config['Custom_Train']['hold_out'] = poly
            config['Train']['data_type'] = f'polysplitnew_{id}_{poly}_t3'
            config['Custom_Train']['data_split_type'] = 'polymer'
            #config['Train']['full_data'] = False
            with open('./config.yaml', 'w') as f:
                _ = yaml.dump(config, f)
                print(f'Updated config for hold out {poly}')
                f.close()
            
            train_models.main()
            
            id += 1
if __name__ == '__main__':
    main()