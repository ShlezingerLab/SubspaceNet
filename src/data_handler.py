"""Subspace-Net 
Details
----------
Name: data_handler.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
"data_handler.py" is aim to handle the creation and processing of synthetic datasets
based on specified parameters and model types.
It includes functions for generating datasets, reading data from files,
computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
Samples (from src.signal_creation): A class for creating samples used in dataset generation.

The script defines the following functions:
* create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
* read_data: Reads data from a file specified by the given path.
* autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
* create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
* create_cov_tensor: Creates a 3D tensor containing the real part, imaginary part, and phase component of the covariance matrix.

"""

# Imports
import torch
import numpy as np
import itertools
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(scenario: str, mode: str, N: int, M: int, T: int,
                    samples_size: float, tau: int, model_type: str,
                    Save:bool=False, dataset_path: str= None,
                    true_doa:list = None, SNR:float = 10, eta: float = 0,
                    geo_noise_var: float = 0, phase:str = None):
    """ Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        scenario (str): "NarrowBand" or "BroadBand" signals.
        mode (str): "coherent" or "non-coherent" nature of signals.
        N (int): number of array sensors.
        M (int): number of sources.
        T (int): number of snapshots.
        samples_size (int): dataset size.
        tau (int): amount of lags for auto-correlation, relevant only for SubspaceNet model.
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
    sampels_model = Samples(scenario= scenario, N= N, M= M,
                    observations=T, freq_values=[0, 500])

    # Generate permutations for CNN model training dataset
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, M):
            doa_permutations.append(list(comb))
    
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        for snr_addition in [0]:
            for i, doa in tqdm(enumerate(doa_permutations)):
                # Samples model creation
                sampels_model.set_doa(doa)             
                # Observations matrix creation
                X = torch.tensor(sampels_model.samples_creation(mode = mode, N_mean= 0,
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
            sampels_model.set_doa(true_doa)
            # Observations matrix creation
            X = torch.tensor(sampels_model.samples_creation(mode = mode, N_mean= 0,
                            N_Var= 1, S_mean= 0, S_Var= 1, SNR= SNR, eta = eta,
                            geo_noise_var = geo_noise_var)[0], dtype=torch.complex64) 
            if model_type.startswith("SubspaceNet"):
                # Generate auto-correlation tensor                                   
                X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            elif model_type.startswith("DeepCNN") and phase.startswith("test"):
                # Generate 3d covariance parameters tensor
                X_model = create_cov_tensor(X)
            else:
                X_model = X
            # Ground-truth creation
            Y = torch.tensor(sampels_model.doa, dtype=torch.float64)
            generic_dataset.append((X,Y))                                                        
            model_dataset.append((X_model,Y))                                                        
    
    if Save:
        torch.save(obj= model_dataset, f=dataset_path + f"/{model_type}_DataSet_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
        torch.save(obj= generic_dataset, f=dataset_path + f"/Generic_DataSet_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
        torch.save(obj= sampels_model, f=dataset_path + f"/Sys_Model_{scenario}_{mode}_{samples_size}_M={M}_N={N}_T={T}_SNR={SNR}_eta={eta}_geo_noise_var{geo_noise_var}" + '.h5')
    
    return model_dataset, generic_dataset, sampels_model

# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert(isinstance(path, (str, Path)))
    data = torch.load(path)
    return data

# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    '''
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

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

# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    '''
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    '''
    Rx_tau = []
    for i in range(tau):
      Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr

# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    '''
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    '''
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx),torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor