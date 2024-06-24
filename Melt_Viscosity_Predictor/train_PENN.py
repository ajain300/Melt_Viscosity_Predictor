from operator import concat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import Enum
from Melt_Viscosity_Predictor.Visc_PENN import Visc_PENN
from pgfingerprinting import fp
from Melt_Viscosity_Predictor.utils.train_utils import assign_sample_ids, polymer_train_test_split
#from Melt_Viscosity_Predictor.utils.train_utils import , custom_train_test_split, assign_sample_ids, hyperparam_opt, hyperparam_opt_ray

import matplotlib.pyplot as plt
from ray import tune

parser = argparse.ArgumentParser(description='Get training vars')
parser.add_argument('--config', default='./config.yaml', help = 'get config file')

class PENN_Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dataset = self._load_data()
        
        # determine test, train splits
        self.train_set, self.test_set = self._test_train_split()

        #
    def _load_data(self) -> pd.DataFrame:
        """
        Load dataset and clean dataframe.
        """
        data = pd.read_excel(args.file)
        data.columns = [str(c) for c in data.columns]
        if args.aug == True and 'level_0' in data.columns:
            data = data.drop(columns = 'level_0')
        data = data[Visc_Data_Feature.data_columns.value]

        # get fingerprints of dataset
        data = self._get_fingerprints(data)
        
        # pull out fp column names
        OG_fp = []
        for c in data.columns:
            if isinstance(c, str):
                if 'fp' in c:
                    OG_fp.append(c)

        # log scale Mw and viscosity
        for c in [Visc_Data_Feature.M_w.value, Visc_Data_Feature.visc.value]:
            data[c] = np.log10(data[c])

        # if PDI is a fault value then impute median of PDI (2.06)
        for i in data.index:
            if not data.loc[i,'PDI'] > 0:
                data.loc[i,'PDI'] = 2.06
            if data.loc[i,'PDI'] > 100:
                data.loc[i,'PDI'] = 2.06
        
        return data

    def _test_train_split(self):
        """
        Split the dataset into train and test with the given arg in the config file
        """
        if args.full_data:
            train_df = self.dataset.sample(frac = 1)
            test_df = self.dataset.sample(frac = 1)
        elif self.args.data_split_type == 'polymer':
            train_df, test_df = polymer_train_test_split(self.dataset, test_size = self.args.test_size, hold_out= self.args.hold_out)
        elif args.data_split_type == 'random':
            train_df, test_df = train_test_split(self.dataset, test_size = args.test_size)
        
        return train_df, test_df

    @staticmethod
    def _fingerprint_blend(smiles_list, conc_list) -> dict:
        """
        Given the SMILES of a blend, calculate the fingerprint.
        TODO assumed a blend with two homopolymers
        """
        
        # Get SMILES for all homopolymers and collect in df
        fp_df_list = []
        for i, smiles in enumerate(smiles_list):
            homo_fp = fp.fingerprint_from_smiles(smiles)
            fp_df_list.append(pd.DataFrame([homo_fp], index = [i]))

        # combine dfs
        combined_df = pd.concat(fp_df_list, axis=0).fillna(0)

        # use the dfs to calculate harmonic sum
        w_1 = conc_list[0]
        w_2 = conc_list[1]
        fp_1 = combined_df.loc[0]
        fp_2 = combined_df.loc[1]
        fp_prod = (fp_1.reset_index(drop = True)*fp_2.reset_index(drop = True)).loc[0,:]
        fp = 1/((w_2*fp_1).add((w_1*fp_2), fill_value = 0).sum())
        fp = (fp_prod*fp).fillna(0)
        
        return fp.loc[0].to_dict()
    
    def _get_fingerprints(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Uses pgfingerprinting to evaluate the fingerprints for all samples in the dataset

        Args:
        data: Dataframe with smiles and weight values

        """
        # Fingerprint homopolymers, copolymers, and blends
        # TODO make this more efficient
        data = data.copy()
        for row in data.index:
            if data.loc[row, Visc_Data_Feature.sample_type.value] == 'Homopolymer':
                data.loc[row, 'FP'] = fp.fingerprint_from_smiles(data.loc[row, Visc_Data_Feature.smiles.value])
            elif data.loc[row, Visc_Data_Feature.sample_type.value] == 'Copolymer':
                smiles_list = [s.strip() for s in data.loc[row, Visc_Data_Feature.smiles.value].split(',')]
                conc_list = [data.loc[row, "Weight 1"], data.loc[row, "Weight 2"]]
                data.loc[row, 'FP'] = fp.fingerprint_any_polymer(smiles_list=smiles_list, conc_list=conc_list)
            elif data.loc[row, Visc_Data_Feature.sample_type.value] == 'Blend':
                smiles_list = [s.strip() for s in data.loc[row, Visc_Data_Feature.smiles.value].split(',')]
                conc_list = [data.loc[row, "Weight 1"], data.loc[row, "Weight 2"]]
                data.loc[row, 'FP'] = self._fingerprint_blend(smiles_list=smiles_list, conc_list=conc_list)
        
        # Normalize the fp dictionary column and fill NaNs with 0
        # Use .apply(pd.Series) for each row, and fill NaNs with 0
        fp_df = data['FP'].apply(pd.Series).fillna(0)

        # Concatenate the fps back with the original data, excluding the original dictionary column
        result_df = pd.concat([data.drop(columns=['Attributes']), fp_df], axis=1)

        return result_df

    def _setup_dataloaders(self, data: pd.DataFrame) -> DataLoader:
        pass

class Visc_Data_Feature(Enum):
    M_w, temp, shear_rate, visc, PDI, polymer, shear_bool, zero_shear_bool, sample_type, smiles, sample_id, w_1, w_2 = 'Mw', 'Temperature', 'Shear_Rate', \
        'Melt_Viscosity', 'PDI', 'Polymer', 'SHEAR', 'ZERO_SHEAR', 'Sample_Type', 'SMILES', 'SAMPLE_ID', 'Weight 1', 'Weight 2'
    data_columns = ['Mw', 'Temperature', 'Shear_Rate', 'Melt_Viscosity', 'PDI', 'Polymer', 'Sample_Type', 'SMILES', 'SAMPLE_ID', 'Weight 1', 'Weight 2']
    

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    #USE GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # get data from excel
    data = pd.read_excel(args.file)
    data.columns = [str(c) for c in data.columns]
    if args.aug == True and 'level_0' in data.columns:
        data = data.drop(columns = 'level_0')
    
    ids = {'shear': [], 'Mw': []}
    data, ids['shear'], ids['Mw'] = assign_sample_ids(data.copy())

    
    OG_fp = []
    for c in data.columns:
        if isinstance(c, str):
            if 'fp' in c:
                OG_fp.append(c)


    #Data Processing #############################################
    if args.do_pca:
        data, fp_cols, pca = fp_PCA(data, 17, cols = OG_fp)
        cols = fp_cols + ['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity', 'PDI']
    else:
        fp_cols = OG_fp
        cols = fp_cols + ['Mw', 'Temperature', 'Shear_Rate','Melt_Viscosity', 'PDI', 'Polymer', 'SHEAR', 'ZERO_SHEAR', 'Sample_Type', 'SMILES', 'SAMPLE_ID']
    
    for c in ['Mw', 'Melt_Viscosity']:
        data[c] = np.log10(data[c])

    data['ZERO_SHEAR'] = 1
    data['SHEAR'] = 0
    data['log_Shear_Rate'] = 0
    for i in data.index:
        if data.loc[i, 'Shear_Rate'] != 0:
            data.loc[i,'log_Shear_Rate'] = np.log10(data.loc[i, 'Shear_Rate'])
            data.loc[i, 'SHEAR'] = 1
            data.loc[i, 'ZERO_SHEAR'] = 0
        if not data.loc[i,'PDI'] > 0:
            data.loc[i,'PDI'] = 2
        if data.loc[i,'PDI'] > 100:
            data.loc[i,'PDI'] = 2
            #data = data.drop([i])

    #################################################################

    #get filtered data
    filtered_data = data.loc[:, cols].dropna(subset = ['Mw', 'Shear_Rate'])

    #Create Test-Train Split#########################################
    if args.full_data:
        train_df = filtered_data.sample(frac = 1)
        test_df = filtered_data.sample(frac = 1) #dummy df for compatibility
    else:
        if args.load_data:
            train_df = pd.read_pickle(f'MODELS/{args.data_date}_{args.data_type}/train_data.pkl')
            if args.data_type == 'full':
                test_df = filtered_data.sample(frac = 0.05) #dummy df for compatibility
            else:
                test_df = pd.read_pickle(f'MODELS/{args.data_date}_{args.data_type}/test_data.pkl')
        else: 
            if args.custom_data_split:
                total_samps = len(ids['Mw']) + len(ids['shear'])
                leave_out_id = [0]
                # Get list of already completed test ids
                completed_ids = [float(f.split('[', 1)[1].split(']')[0]) for f in os.listdir('./MODELS') if 'custom' in f]
                print(f'comp ids {completed_ids}')
                while (data.loc[data['SAMPLE_ID'] == leave_out_id[0]].shape[0] < 5 or data.loc[data['SAMPLE_ID'] == leave_out_id[0]].shape[0] > 10) and leave_out_id[0] not in completed_ids:
                    leave_out_id = random.sample(ids[args.leave_out],1)
                    print(data.loc[data['SAMPLE_ID'] == leave_out_id[0]])
                train_df, test_df = custom_train_test_split(filtered_data, test_id = leave_out_id, id_col= 'SAMPLE_ID')
                args.data_type = f'{args.data_type}_{args.leave_out}_{leave_out_id}'
            
            elif args.data_split_type == 'polymer':
                train_df, test_df = polymer_train_test_split(filtered_data, test_size = args.test_size, hold_out= args.hold_out)
                print('SHEAR_TEST_post split', np.max(test_df['Shear_Rate']), np.min(test_df['Shear_Rate']))
            elif args.data_split_type == 'random':
                train_df, test_df = train_test_split(filtered_data, test_size = args.test_size)
            else:
                train_df, test_df = train_test_split(filtered_data, test_size = args.test_size)
            
            train_df = train_df.loc[:, (train_df != 0).any(axis=0)]
            new_fp = []
            for c in train_df.columns:
                if isinstance(c, str):
                    if 'fp' in c:
                        new_fp.append(c)

            if len(OG_fp) != len(new_fp):
                test_df = test_df.drop(columns = list(set(OG_fp) - set(new_fp)))
    ####################################################################################
    print('train size = ' + str(len(train_df)))
    print('test size = ' + str(len(test_df)))

    #Scale Variables #################################################################
    #Only fit scaling on train set and not the test set
    logMw = np.array(train_df['Mw']).reshape((-1,1))
    shear = np.array(train_df['Shear_Rate']).reshape((-1,1))
    Temp = np.array(train_df['Temperature']).reshape((-1,1))
    Temp = Temp+273.15
    PDI = np.array(train_df['PDI']).reshape((-1,1))

    scaler = MinMaxScaler(copy = False)
    XX = torch.tensor(np.array(scaler.fit(train_df.filter(fp_cols)).transform(train_df.filter(fp_cols))), dtype=torch.float64).to(device)
    yy = np.array(train_df.loc[:,'Melt_Viscosity']).reshape((-1,1))

    y_scaler = MinMaxScaler().fit(yy)
    yy = y_scaler.transform(yy);
    yy = torch.tensor(yy, dtype=torch.float64).to(device)
    T_scaler = MinMaxScaler().fit(Temp)
    T = T_scaler.transform(Temp);
    T = torch.tensor(T, dtype=torch.float64).to(device)
    M_scaler = MinMaxScaler().fit(logMw)
    M = y_scaler.transform(logMw);
    print(f'min M {min(M)}')
    print(f'min Mw {min(logMw)}')
    M = torch.tensor(M, dtype=torch.float64).to(device)
    S_trans = PowerTransformer(standardize = False).fit(shear)
    S_scaler = MinMaxScaler().fit(S_trans.transform(shear))
    S = S_scaler.transform(S_trans.transform(shear))
    S = torch.tensor(S, dtype=torch.float64).to(device)
    P_scaler = MinMaxScaler().fit(PDI)
    P = P_scaler.transform(PDI)
    P = torch.tensor(P, dtype=torch.float64).to(device)
    #shear = S_scaler.transform((shear))
    #gpr_Mcr, mcr_cv_error = Mcr_gpr_train(OG_fp, None, M_scaler, scaler, transform = False)
    
    y_test = y_scaler.transform(np.array(test_df.loc[:,'Melt_Viscosity']).reshape((-1,1)))
    y_test = torch.tensor(y_test, dtype=torch.float64).to(device)
    X_test = np.array(scaler.transform(test_df.filter(fp_cols)))
    X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
    M_test = y_scaler.transform(np.array(test_df['Mw']).reshape((-1,1)))
    M_test = torch.tensor(M_test, dtype=torch.float64).to(device)
    S_test = S_scaler.transform(S_trans.transform(np.array(test_df['Shear_Rate']).reshape((-1,1))))
    S_test = torch.tensor(S_test, dtype=torch.float64).to(device)
    T_test = np.array(test_df['Temperature']).reshape((-1,1))
    T_test = T_scaler.transform(T_test+273.15)
    T_test = torch.tensor(T_test, dtype=torch.float64).to(device)
    P_test = P_scaler.transform(np.array(test_df['PDI']).reshape((-1,1)))
    P_test = torch.tensor(P_test, dtype=torch.float64).to(device)
    #################################################################################

    train_data = MVDataset(XX.float(), M.float(), S.float(), T.float(), P.float(), yy.float())
    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    
    test_data = MVDataset(X_test.float(), M_test.float(), S_test.float(), T_test.float(), P_test.float(), y_test.float())
    test_loader = DataLoader(test_data, batch_size = 32, shuffle = True)
    
    
    #print([visc for idx, (XX, M, S, T, P, visc) in enumerate(train_loader)])

    #####INITIALIZE MODEL AND TRAINING VARIABLES######
    EPOCHS = 200
    model, train_loss, val_loss = run_training({"l1": 120, "l2": 120, "d1":0.2, "d2":0.2}, n_fp = XX.shape[1], train_loader = train_loader, test_loader = test_loader, device = device, EPOCHS = EPOCHS)
    
    print(type(model))
    plt.figure()
    plt.plot(range(EPOCHS),train_loss)
    plt.plot(range(EPOCHS),val_loss)

    plt.savefig('learning_curve.png')

    plt.figure()
    train_pred = model(XX.float(), M.float(), S.float(), T.float(), P.float()).cpu().detach().numpy()
    test_pred = model(X_test.float(), M_test.float(), S_test.float(), T_test.float(), P_test.float()).cpu().detach().numpy()
    plt.scatter(yy.cpu(), train_pred, c = 'tab:blue',label = f'Train: MSELoss = {train_loss[0]}')
    plt.scatter(y_test.cpu(), test_pred, c = 'tab:orange' ,label = f'Test: MSELoss = {train_loss[0]}')
    plt.legend()
    plt.savefig('parity.png')
    
    torch.save(model, 'MODELS/HyperNet.pt')

class MVDataset(Dataset):
    """
    Dataset format for PENN.
    """
    def __init__(self, FP, M, S, T, PDI, Visc):
        self.FP = FP
        self.M = M
        self.S = S
        self.T = T
        self.PDI = PDI
        self.Visc = Visc

    def __len__(self):
        return len(self.Visc)
    
    def __getitem__(self,idx):
        return self.FP[idx], self.M[idx], self.S[idx], self.T[idx], self.PDI[idx], self.Visc[idx]
    
if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

