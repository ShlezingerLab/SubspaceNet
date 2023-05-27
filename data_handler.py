import torch
import numpy as np
from System_Model import System_model
from Signal_creation import Samples
from tqdm import tqdm
import random
import h5py
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(scenario: str, mode: str, N: int, M: int, T: int,
                    samples_size: float, tau: int, model_type: str,
                    Save:bool=False, dataset_path: str= None,
                    true_doa:list = None, SNR:float = 10, eta: float = 0,
                    geo_noise_var: float = 0, phase:str = None):
    """function for generating synthetic dataset,
        each is applied for the desired model_type input.

    Args:
    -----
    scenario (str): "NarrowBand" or "BroadBand" signals.
    mode (str): "coherent" or "non-coherent" nature of signals.
    N (int): number of array sensors.
    M (int): number of sources.
    T (int): number of snapshots.
    samples_size (int): dataset size.
    tau (int): amount of lags for auto-correlation, relevant only for SubspaceNet model..
    model_type (str): the model type.
    Save (bool, optional): wether or not save dataset. Defaults to False.
    dataset_path (str, optional): path for saving dataset. Defaults to None.
    true_doa (list, optional): predefined angels. Defaults to None.
    SNR (float, optional): SNR for samples creations. Defaults to 10.
    eta (float, optional): sensors distance deviation from ideal array,
                    relevant for array-mismatches scenarios. Defaults to 0.
    geo_noise_var (float, optional): steering vector added noise,
                    relevant for array-mismatches scenarios. Defaults to 0.
    phase (str, optional): test or training phase for dataset,
                    relevant only for CNN model. Defaults to None.

    Returns:
    --------
    tuple: the desired dataset comprised from (X-samples, Y-labels)
    """ 
    generic_dataset = []
    model_dataset = []
    system_model = Samples(scenario= scenario, N= N, M= M,
                    observations=T, freq_values=[0, 500])

    # Generate permutations for CNN model training dataset
    if model_type.startswith("CNN") and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, M):
            doa_permutations.append(list(comb))
    
    if model_type.startswith("CNN") and phase.startswith("train"):
        for snr_addition in [0, 1]:
            for i, doa in tqdm(enumerate(doa_permutations)):
                # Samples model creation
                system_model.set_doa(doa)             
                # Observations matrix creation
                X = torch.tensor(system_model.samples_creation(mode = mode, N_mean= 0,
                                N_Var= 1, S_mean= 0, S_Var= 1, SNR= SNR + snr_addition, eta = eta,
                                geo_noise_var = geo_noise_var)[0], dtype=torch.complex64)
                X_model = create_cov_tensor(X)
                # Ground-truth creation  
                Y = torch.zeros_like(torch.tensor(angles_grid))
                for angle in doa:
                    Y[list(angles_grid).index(angle)] = 1
                model_dataset.append((X_model,Y))
                generic_dataset.append((X,Y))
    else:
        for i in tqdm(range(samples_size)):
            # Samples model creation      
            # system_model = Samples(scenario= scenario, N= N, M= M,
            #                 DOA= true_doa, observations=T, freq_values=[0, 500])
            system_model.set_doa(true_doa)
            # Observations matrix creation
            X = torch.tensor(system_model.samples_creation(mode = mode, N_mean= 0,
                            N_Var= 1, S_mean= 0, S_Var= 1, SNR= SNR, eta = eta,
                            geo_noise_var = geo_noise_var)[0], dtype=torch.complex64) 
            if model_type.startswith("SubspaceNet"):
                # Generate auto-correlation tensor                                   
                X_model = create_autocorr(X, tau).to(torch.float)
            elif model_type.startswith("CNN") and phase.startswith("test"):
                # Generate 3d covariance parameters tensor
                X_model = create_cov_tensor(X)
            else:
                X_model = X
            # Ground-truth creation
            Y = torch.tensor(system_model.DOA, dtype=torch.float64)
            generic_dataset.append((X,Y))                                                        
            model_dataset.append((X_model,Y))                                                        
    
    if Save:
        torch.save(obj= model_dataset, f=dataset_path + f"/{model_type}_DataSet_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
        torch.save(obj= generic_dataset, f=dataset_path + f"/Generic_DataSet_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
        torch.save(obj= system_model, f=dataset_path + f"/Sys_Model_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
    
    return model_dataset, generic_dataset, system_model

def Read_Data(Data_path):
    Data = torch.load(Data_path)
    return Data

def autocorr_matrix(X, lag):
    '''
    This function compute the autocorrelation matrices of the T samples
    x(t), t=1, ... ,T for a given lag.
    ------- Input ------
    X(input): Samples matrix input shape [N, T] 
    lag(input) = the requested delay of the Autocorrelation calculation
    Rx_lag = the autocorrelation matrix in a given lag []
    
    Returns:
    --------
    
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
    
    Args:
    ------
    X: observation matrix input, from size (BS, N, T)
    tau: maximal time difference for auto-correlation tensor
    
    Returns:
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
                          Save=False, dataset_path= None, true_doa = 0,
                          SNR = 10, eta = 0, geo_noise_var = 0)