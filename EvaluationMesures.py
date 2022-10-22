import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");


def permute_prediction(prediction):
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p),dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class PRMSELoss(nn.Module):
    def __init__(self):
        super(PRMSELoss, self).__init__()
    def forward(self, preds, DOA):
      prmse = []
      for iter in range(preds.shape[0]):
          prmse_list = []
          Batch_preds = preds[iter].to(device)
          targets = DOA[iter].to(device)
          prediction_perm = permute_prediction(Batch_preds).to(device)
          for prediction in prediction_perm:
              ## Old evaluation measure [-pi/2, pi/2] 
              error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2                        # Calculate error with modulo pi
              prmse_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)                          # Calculate MSE 
              prmse_list.append(prmse_val)
          prmse_tensor = torch.stack(prmse_list, dim = 0)
          prmse_min = torch.min(prmse_tensor)
          prmse.append(prmse_min)
      result = torch.sum(torch.stack(prmse, dim = 0))
      return result
class PMSELoss(nn.Module):
    def __init__(self):
        super(PMSELoss, self).__init__()
    def forward(self, preds, DOA):
      prmse = []
      for iter in range(preds.shape[0]):
          prmse_list = []
          Batch_preds = preds[iter].to(device)
          targets = DOA[iter].to(device)
          prediction_perm = permute_prediction(Batch_preds).to(device)
          for prediction in prediction_perm:
              ## Old evaluation measure [-pi/2, pi/2] 
              error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2                        # Calculate error with modulo pi
              prmse_val = (1 / len(targets)) * (torch.linalg.norm(error) ** 2)                           # Calculate MSE 
              prmse_list.append(prmse_val)
          prmse_tensor = torch.stack(prmse_list, dim = 0)
          prmse_min = torch.min(prmse_tensor)
          prmse.append(prmse_min)
      result = torch.sum(torch.stack(prmse, dim = 0))
      return result