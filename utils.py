import numpy as np
import torch
import random

def sum_of_diag(Matrix):
    coeff = []
    diag_index = np.linspace(-Matrix.shape[0] + 1, Matrix.shape[0] + 1, 2 * Matrix.shape[0] - 1, endpoint = False, dtype = int)
    for idx in diag_index:
        coeff.append(np.sum(Matrix.diagonal(idx)))
    return coeff
    
def find_roots(coeff):
    coeff = np.array(coeff)
    A = np.diag(np.ones((len(coeff)-2,), coeff.dtype), -1)
    A[0,:] = -coeff[1:] / coeff[0]
    roots = np.array(np.linalg.eigvals(A))
    return roots

def Set_Overall_Seed(SeedNumber = 42):
    random.seed(SeedNumber)
    np.random.seed(SeedNumber)
    torch.manual_seed(SeedNumber)