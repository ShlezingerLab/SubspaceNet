import torch
import numpy as np
from System_Model import System_model
from Signal_creation import Samples
from tqdm import tqdm
import random
import h5py
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(scenario, mode, N, M, T, Sampels_size, tau,
                          Save=False, DataSet_path= None, True_DOA = None,
                          SNR = 10, eta = 0, geo_noise_var = 0):
    '''
    @Scenario = "NarrowBand" or "BroadBand"
    @mode = "coherent", "non-coherent"
    '''
    DataSet = []
    DataSetRx = []
    print("Updated")
    for i in tqdm(range(Sampels_size)):
        ####### Model Initialization #######                   
        Sys_Model = Samples(scenario= scenario, N= N, M= M,
                            DOA= True_DOA, observations=T, freq_values=[0, 500])    # Samples model creation
        X = torch.tensor(Sys_Model.samples_creation(mode = mode,                    # Observations matrix creation
                                                    N_mean= 0, N_Var= 1,
                                                    S_mean= 0, S_Var= 1,
                                                    SNR= SNR, eta = eta,
                                                    geo_noise_var = geo_noise_var)[0],
                         dtype=torch.complex64)                                     
        Y = torch.tensor(Sys_Model.DOA, dtype=torch.float64)                        # DoA vector
        DataSet.append((X,Y))                                                       # Couple observations and DoA's 
        # New_Rx_tau = create_autocorr(X, tau).to(torch.float)                        # Generate auto-correlation tensor
        # DataSetRx.append((New_Rx_tau,Y))                                            # Couple observations and DoA's
    
    if Save:
        torch.save(obj= DataSet  , f=DataSet_path + r"/DataSet_x_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
        # torch.save(obj= DataSetRx, f=DataSet_path + r"/DataSet_Rx_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
        torch.save(obj= Sys_Model, f=DataSet_path + r"/Sys_Model_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
    
    return DataSet, DataSetRx ,Sys_Model

def create_dataset_for_cnn(scenario, mode, N, M, T, Sampels_size, tau,
                          Save=False, DataSet_path= None, True_DOA = None,
                          SNR = 10, eta = 0, geo_noise_var = 0, phase = "train"):
    '''
    @Scenario = "NarrowBand" or "BroadBand"
    @mode = "coherent", "non-coherent"
    '''
    DataSet = []
    DataSetRx = []
    if phase.startswith("train"):
        # generate doa combinations
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, 2):
            doa_permutations.append(list(comb))

        
        for i, doa in tqdm(enumerate(doa_permutations)):
            ####### Model Initialization #######                   
            Sys_Model = Samples(scenario= scenario, N= N, M= M,
                                DOA= doa, observations=T, freq_values=[0, 500])    # Samples model creation
            X = torch.tensor(Sys_Model.samples_creation(mode = mode,                    # Observations matrix creation
                                                        N_mean= 0, N_Var= 1,
                                                        S_mean= 0, S_Var= 1,
                                                        SNR= SNR, eta = eta,
                                                        geo_noise_var = geo_noise_var)[0],
                            dtype=torch.complex64)
            X = create_cov_tensor(X)                                     
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            DataSet.append((X,Y))                                                       # Couple observations and DoA's 
            # New_Rx_tau = create_autocorr(X, tau).to(torch.float)                        # Generate auto-correlation tensor
            # DataSetRx.append((New_Rx_tau,Y))                                            # Couple observations and DoA's
    elif phase.startswith("test"):
        for i in tqdm(range(Sampels_size)):
            ####### Model Initialization #######                   
            Sys_Model = Samples(scenario= scenario, N= N, M= M,
                                DOA= True_DOA, observations=T, freq_values=[0, 500])    # Samples model creation
            X = torch.tensor(Sys_Model.samples_creation(mode = mode,                    # Observations matrix creation
                                                        N_mean= 0, N_Var= 1,
                                                        S_mean= 0, S_Var= 1,
                                                        SNR= SNR, eta = eta,
                                                        geo_noise_var = geo_noise_var)[0],
                            dtype=torch.complex64)
            X = create_cov_tensor(X)                                     
            Y = torch.tensor(Sys_Model.DOA, dtype=torch.float64)                        # DoA vector
            DataSet.append((X,Y))                                                       # Couple observations and DoA's 
    
    if Save:
        torch.save(obj= DataSet  , f=DataSet_path + r"/DataSet_cnn_x_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
        # torch.save(obj= DataSetRx, f=DataSet_path + r"/DataSet_Rx_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
        torch.save(obj= Sys_Model, f=DataSet_path + r"/Sys_Model_cnn_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}".format(scenario, mode, Sampels_size, M, N, T, SNR, eta, geo_noise_var) + '.h5')
    
    return DataSet, DataSetRx ,Sys_Model

def Read_Data(Data_path):
    Data = torch.load(Data_path)
    return Data

def autocorr_matrix(X, lag):
    '''
    This function compute the autocorrelation matrices of the T samples
    x(t), t=1, ... ,T for a given lag.
    ------- Input ------
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

def create_autocorr(X, tau):
    '''
    This function returns a tensor contains all
    the autocorrelation tau matrices, for lag 0 to tau
    {R_x(lag)}, lag  = 0, ..., tau
    
    Input:
    --------------------------------------------
    X: observation matrix input, from size (BS, N, T)
    tau: maximal time difference for auto-correlation tensor
    
    Output:
    --------------------------------------------
    Rx_autocorr: Tensor contains all the auto-correlation matrices,
                 from size (Batch size, tau, 2N, N)
    '''
    Rx_tau = []
    for i in range(tau):
      Rx_tau.append(autocorr_matrix(X, lag = i))
    Rx_autocorr = torch.stack(Rx_tau, dim = 0)
    return Rx_autocorr

def create_cov_tensor(X):
    '''
    Creating a 3D tensor (NxNx3) contain:
    1. Rx real part
    2. Rx imaginary part
    3. Rx phase component    
    Args:
    -----
    X: observation matrix input, size: (N, T)
    
    Returns:
    --------
    Rx_tensor: Tensor contains all the auto-correlation matrices,
                 from size (Batch size, N, N, 3)
    '''
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx),torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor

if __name__ == "__main__":
    tau = 8                     # Number of lags
    N = 8                       # Number of sensors
    M = 2                       # number of sources
    T = 100                       # Number of observations, ideal = 200 or above
    SNR = 10                   # Signal to noise ratio, ideal = 10 or above
    
    ## Signal parameters
    scenario = "NarrowBand"     # signals type, options: "NarrowBand", "Broadband_OFDM", "Broadband_simple"
    mode = "coherent"           # signals nature, options: "non-coherent", "coherent"
    
    ## Array mis-calibration values
    eta = 0                     # Deviation from sensor location, normalized by wavelength, ideal = 0
    geo_noise_var = 0           # Added noise for sensors response
    
    # simulation parameters
    samples_size = 70000        # Overall dateset size 
    train_test_ratio = 0.05    # training and testing datasets ratio 
    create_dataset_for_cnn(scenario, mode, N, M, T, 1, tau,
                          Save=False, DataSet_path= None, True_DOA = 0,
                          SNR = 10, eta = 0, geo_noise_var = 0)