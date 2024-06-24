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
import matplotlib.pyplot as plt
from ray import tune
from enum import Enum
import os
import wandb



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

class LossTypes(Enum):
    tot_train = "epoch_train_loss"
    avg_train = "avg_train_loss"
    tot_val = "epoch_validation_loss"
    avg_val = "avg_validation_loss"
    a1 = "a1_loss"
    a2 = "a2_loss"
    phys = "physics_informed_loss"

    """
    The `run_training` function trains a model using the specified configuration and data loaders, with
    options for checkpointing, early stopping, and hyperparameter tuning.
    
    :param config: Contains configuration settings for the training
    process, such as hyperparameters, model architecture details, and optimization settings. It could
    include values like learning rate, batch size, weight decay, and other parameters that affect the
    training of the model
    :param n_fp: Fingerprint dimension
    :param train_loader: The `train_loader` parameter in the `run_training` function is used to pass the
    data loader for training the model. It typically contains the training dataset that is used to train
    the model in each epoch. The data loader provides batches of data to the model during training
    :param test_loader: The `test_loader` parameter in the `run_training` function is used to provide
    the data loader for the validation/testing dataset. This data loader is used during each epoch to
    evaluate the model's performance on the validation/testing set and calculate the validation loss
    :param device: The `device` parameter in the `run_training` function is used to specify whether to
    use a GPU (cuda) if available or fallback to CPU if not. It is set to `torch.device("cuda:0" if
    torch.cuda.is_available() else "cpu")` which dynamically checks if
    :param EPOCHS: The `EPOCHS` parameter in the `run_training` function represents the number of times
    the model will iterate over the entire training dataset during the training process. It determines
    how many times the model will update its weights based on the training data to minimize the loss
    function, defaults to 100 (optional)
    :param tuning: The `tuning` parameter in the `run_training` function is a boolean flag that
    indicates whether the training process is part of hyperparameter tuning or not. When `tuning` is set
    to `True`, the function will report the validation loss for tuning purposes. This flag is used to
    control, defaults to False (optional)
    :param model_arch: The `model_arch` parameter in the `run_training` function is expected to be a
    model architecture class or function that will be used to instantiate the model for training. This
    architecture should be provided as an argument when calling the `run_training` function. It is
    essential for defining the neural network model
    :param logger: The `logger` parameter in the `run_training` function is used to provide a logging
    interface for recording important information and messages during the training process. It is
    essential for tracking the progress of the training, logging metrics, errors, and other relevant
    details for debugging and analysis purposes. The logger helps in
    :return: The function `run_training` returns the trained model if it has already been trained or if
    the training is complete. If the training is not complete, it returns the model after training it
    for the specified number of epochs.
    """
def run_training(config, n_fp = None, train_loader = None, 
                test_loader = None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                EPOCHS = 100, tuning = False, model_arch = None, logger = None,**kwargs):
    assert model_arch, "Please provide a model architecture to train on."
    assert logger, "Please provide a logger"
    apply_gradnorm = kwargs.get("apply_gradnorm", False)
    run = kwargs.get("run", None)
    lr = kwargs.get("lr", 1e-6)
    fold = kwargs.get("fold", None)
    logger.info(f"Learning rate at {lr}")
    
    model = model_arch(n_fp, config,device,apply_gradnorm = apply_gradnorm, run = run, fold = kwargs.get("fold", None)).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 20, factor = kwargs.get("reduce_lr_factor", 0.1))
    criterion = nn.MSELoss()
    start_epoch = 0
    
    # If this is not hp tuning and a ckpt folder is given then create a checkpoint path and load if the ckpt exists
    if not tuning:
        ckpt_folder = kwargs.get("ckpt_folder", False)
        if ckpt_folder:
            ckpt_path = os.path.join(ckpt_folder, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                start_epoch, optimizer, scheduler = model.load_checkpoint(ckpt_path, optimizer, scheduler)
                logger.info(f"Model loaded at epoch {start_epoch}.")

    # if the model is already trained then return the trained model
    if start_epoch == EPOCHS-1 or model.training_complete == True:
        logger.info("Already trained model.")
        return model

    train_loss_list, val_loss_list = [], []
    for epoch_idx in range(start_epoch, EPOCHS):
        torch.autograd.set_detect_anomaly(False)
        train_loss_dict = model.train_model(train_loader, optimizer, criterion,config, **kwargs)
        # train_loss, avg_train_loss, avg_a1, avg_a2 = model.train(train_loader, optimizer, criterion, config = config, device = device, **kwargs)

        val_loss, avg_val_loss = model.evaluate(test_loader, criterion)
        scheduler.step(avg_val_loss)
        #scheduler.step(train_loss_dict[LossTypes.avg_train.value])

        if run:
            run.log({f"sch_learning_rate_fold{fold}":optimizer.param_groups[0]['lr'], f"avg_val_loss_fold{fold}":avg_val_loss})
        # train_loss_list.append(train_loss_dict[LossTypes.tot_train.value].cpu().item())
        # val_loss_list.append(val_loss.cpu().item())
        
        if (epoch_idx%25 == 0 or epoch_idx == EPOCHS - 1) and not tuning:
            logger.info("-----------------------------------")
            logger.info("Epoch %d" % (epoch_idx+1))
            logger.info("-----------------------------------")

            logger.info("Training Loss: %.4f. Validation Loss: %.4f. " % (train_loss_dict[LossTypes.avg_train.value], avg_val_loss))
            logger.info(f"Train Losses: {train_loss_dict}")

            if ckpt_folder:
                model.save_checkpoint(ckpt_path, optimizer, scheduler, epoch_idx)

        if tuning:
            #tune.report(loss = 1)
            logger.info('val_loss')
            logger.info(avg_val_loss)
            tune.report(loss=float(avg_val_loss.cpu().detach().numpy()))
        else:
            # If not tuning then check early stopping
            model.early_stopping(val_loss)
            if model.early_stopping.early_stop:
                print("Early stopping at epoch:", epoch_idx +1)
                break
        
    if not tuning:
        model.save_checkpoint(ckpt_path, optimizer, scheduler, epoch_idx, complete = True)
        return model

# NOTE deprecated. This function is implemented in the model classes of Visc_PENN
def train(model, dataloader, optimizer, criterion, config, device, scheduler = None, apply_gradnorm = False):
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

# NOTE deprecated. This function is implemented in the model classes of Visc_PENN
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