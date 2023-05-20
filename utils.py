"""Subspace-Net 
Details
----------
Name: utils.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This script defines some helpful functions:
    * sum_of_diag: returns the some of each diagonal in a given matrix 
    * find_roots: solves polynomial equation defines by polynomial coefficients 
    * set_unified_seed: Sets unified seed for all random attributed in the simulation
"""

import numpy as np
import torch
import random

def sum_of_diag(matrix:np.ndarray) -> list:
    coeff = []
    diag_index = np.linspace(-matrix.shape[0] + 1, matrix.shape[0] + 1, 2 * matrix.shape[0] - 1, endpoint = False, dtype = int)
    for idx in diag_index:
        coeff.append(np.sum(matrix.diagonal(idx)))
    return coeff
    
def find_roots(coeff):
    coeff = np.array(coeff)
    A = np.diag(np.ones((len(coeff)-2,), coeff.dtype), -1)
    A[0,:] = -coeff[1:] / coeff[0]
    roots = np.array(np.linalg.eigvals(A))
    return roots
    
def set_unified_seed(seed:int = 42):
    """Sets unified seed for all random attributed in the simulation
    
    Args:
        seed (int, optional): seed value. Defaults to 42.
    """    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_k_angles(grid_size, k:int, prediction):
    angels_grid = torch.linspace(-90, 90, grid_size)
    doa_prediction = angels_grid[torch.topk(prediction.flatten(), k).indices]
    return doa_prediction