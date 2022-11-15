import torch
import numpy as np
from System_Model import System_model
from Signal_creation import Samples
from tqdm import tqdm
import random
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def CreateDataSetCombined(scenario, mode, N, M, T, Sampels_size, tau, Save=False, DataSet_path= None, True_DOA = None, SNR = 10):
    '''
    @Scenario = "NarrowBand" or "BroadBand"
    @mode = "coherent", "non-coherent"
    '''
    DataSet = []
    DataSetRx = []
    print("Updated")
    for i in tqdm(range(Sampels_size)):
        # # System Model Initialization
        # Sys_Model = System_model(scenario= scenario, N= N, M= M)
        
        # # Samples Creation - Model Initialization                           
        # sys_model_samples = Samples(Sys_Model, DOA= True_DOA, observations=T)                 
        
        # Samples Creation - Model Initialization                           
        
        Sys_Model = Samples(scenario= scenario, N= N, M= M, DOA= True_DOA, observations=T, freq_values=[0, 100])                 
        X = torch.tensor(Sys_Model.samples_creation(mode = mode, N_mean= 0,
                                                    N_Var= 1, S_mean= 0, S_Var= 1,
                                                    SNR= SNR)[0], dtype=torch.complex64)                   # Samples Creation 
        Y = torch.tensor(Sys_Model.DOA, dtype=torch.float64)
        DataSet.append((X,Y))
        
        New_Rx_tau = Create_Autocorr_tensor_for_data_loader(X, tau).to(torch.float)
        DataSetRx.append((New_Rx_tau,Y))
    
    if Save:
        torch.save(obj= DataSet, f=DataSet_path + '/DataSet_x_{}_{}_{}_M={}_N={}_T={}_SNR={}'.format(scenario, mode, Sampels_size, M, N, T, SNR) + '.h5')
        torch.save(obj= DataSetRx, f=DataSet_path + '/DataSet_Rx_{}_{}_{}_M={}_N={}_T={}_SNR={}'.format(scenario, mode, Sampels_size, M, N, T, SNR) + '.h5')
        torch.save(obj= Sys_Model, f=DataSet_path + '/Sys_Model_{}_{}_{}_M={}_N={}_T={}_SNR={}'.format(scenario, mode, Sampels_size, M, N, T, SNR) + '.h5')
    
    return DataSet, DataSetRx ,Sys_Model

def Read_Data(Data_path):
    Data = torch.load(Data_path)
    return Data

def torch_Autocorr_mat_for_data_loader(X, lag):
    '''
    This function compute the autocorrelation matrices of the T samples
    x(t), t=1, ... ,T for a given lag.
    @ X(input) = Samples matrix input shape [N, T] 
    @ lag(input) = the requested delay of the Autocorrelation calculation
    @ Rx_lag = the autocorrelation matrix in a given lag []
    '''
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128).to(device)
    for t in range(X.shape[1] - lag):
        # meu = torch.mean(X,1)
        x1 = torch.unsqueeze(X[:, t], 1).to(device)
        x2 = torch.t(torch.unsqueeze(torch.conj(X[:, t + lag]),1)).to(device)
        Rx_lag += torch.matmul(x1 - torch.mean(X), x2 - torch.mean(X)).to(device)
    Rx_lag = Rx_lag / (X.shape[-1] - lag)
    Rx_lag = torch.cat((torch.real(Rx_lag),torch.imag(Rx_lag)), 0)
    return Rx_lag

def Create_Autocorr_tensor_for_data_loader(X, tau):
    '''
    This function Returns a Tensor which contain all the Autocorrelation
    matrices for lags equal to 0 until tau
    {R_x(lag)}, where lag  = 0, ..., tau
    @ X(input) - Sampels matrix input shape [BS, N, T]
    @ tau(input) - the highest delay requested for the Autocorrelation Tensor 
    (should be greater than number of Observations T)
    @ Rx_Autocorr = Tensor contains all the AutoCorrelation matrices [Batch size, tau, 2N, N]
    '''
    Rx_tau = []
    for i in range(tau):
      Rx_tau.append(torch_Autocorr_mat_for_data_loader(X, lag = i))
    Rx_Autocorr = torch.stack(Rx_tau, dim = 0)
    return Rx_Autocorr