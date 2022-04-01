from sklearn.model_selection import KFold, StratifiedKFold
from random import sample
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
from validation.metrics import MSE, OME
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras_tuner as kt
import json
from os import sys
sys.path.append('../')
from MODELS.ViscNN import predict_all_cv, ViscNN_concat_HP
import time
from math import floor
import yaml
import train_models


def main():
    test_sizes = list(1 - np.linspace(0.1, 0.9, 9))
    print(test_sizes)
    train_error = []
    test_error = []
    
    for trial in [2,3,4]:
        for te_sz in test_sizes:
            print(te_sz)
            with open('./config.yaml') as f:
                config = yaml.safe_load(f)
                f.close()
            
            config['Train']['test_size'] = round(float(te_sz),1)
            config['Train']['data_type'] = f'LC{trial}_split_test_{str(round(float(te_sz), 1))}'
            
            with open('./config.yaml', 'w') as f:
                _ = yaml.dump(config, f)
                print(f'Updated config for test {te_sz}')
                f.close()
            train_models.main()


if __name__ == '__main__':
    main()