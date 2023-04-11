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
#from validation.tests import assign_sample_ids
from Visc_PIMI import Visc_PIMI
import matplotlib.pyplot as plt
from ray import tune


parser = argparse.ArgumentParser(description='Get training vars')
parser.add_argument('--config', default='./config.yaml', help = 'get config file')

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
    #data, ids['shear'], ids['Mw'] = assign_sample_ids(data.copy())

    
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


def run_training1(config):
    tune.report(loss = 1)


def run_training(config, n_fp = None, train_loader = None, test_loader = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), EPOCHS = 100, tuning = False):
    model = Visc_PIMI(n_fp, config,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()

    train_loss_list, val_loss_list = [], []
    for epoch_idx in range(EPOCHS):
        torch.autograd.set_detect_anomaly(False)
        train_loss, avg_train_loss, avg_a1, avg_a2 = train(model, train_loader, optimizer, criterion, config = config, device = device)
        scheduler.step(train_loss)
        if not train_loss > -2:
            print(train_loss)
            break
        val_loss, avg_val_loss = evaluate(model, test_loader, criterion)

        train_loss_list.append(avg_train_loss.cpu().item())
        val_loss_list.append(avg_val_loss.cpu().item())
        
        if (epoch_idx%25 == 0 or epoch_idx == EPOCHS - 1) and not tuning:
            print("-----------------------------------")
            print("Epoch %d" % (epoch_idx+1))
            print("-----------------------------------")

            print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
            print("a1 loss: %.4f. a2 loss: %.4f. " % (avg_a1, avg_a2))

        if tuning:
            #tune.report(loss = 1)
            print('val_loss')
            print(avg_val_loss)
            tune.report(loss=float(avg_val_loss.cpu().detach().numpy()))

    if not tuning:
        return model, train_loss_list, val_loss_list


def train(model, dataloader, optimizer, criterion, config, device, scheduler = None):

    model.train()

    # Record total loss
    total_loss = 0.
    # Get the progress bar for later modification


    # Mini-batch training
    for batch_idx, (XX, M, S, T, P, visc) in enumerate(dataloader):
        # print(type(data))
        # print(len(data))
        # print("label shape",data[1].shape)
        # print("input shape",data[0].shape)
        #with torch.autograd.detect_anomaly():
            
        out, a1,a2 = model(XX, M, S, T, P, train = True)
        if torch.isnan(out).any():
            # print('invalid out')
            # nan_idx = (torch.isnan(out)==1).nonzero(as_tuple=True)
            # print(M[nan_idx], S[nan_idx], T[nan_idx])
            break
        optimizer.zero_grad()

        loss = criterion(out, visc)
        a1_loss = config["a_weight"]*criterion(a1, torch.ones_like(a1).to(device)*torch.tensor(1).to(device))
        a2_loss = config["a_weight"]*criterion(a2, torch.ones_like(a2)*torch.tensor(3.4).to(device)) 
        loss = loss + a2_loss + a1_loss
        #print(loss.device)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss
        a1_loss+=a1_loss
        a2_loss+=a2_loss
        
    return total_loss, total_loss / len(dataloader), a1_loss/ len(dataloader), a2_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar 
        for batch_idx, (XX, M, S, T, P, visc) in enumerate(dataloader):
        
            with torch.no_grad():
              out = model(XX, M, S, T, P)

            loss = criterion(out, visc)

            total_loss += loss
    
    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

if __name__ == '__main__':
    main()