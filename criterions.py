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

class RMSPELoss(nn.Module):
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, preds, DOA):
      rmspe = []
      for iter in range(preds.shape[0]):
          rmspe_list = []
          Batch_preds = preds[iter].to(device)
          targets = DOA[iter].to(device)
          prediction_perm = permute_prediction(Batch_preds).to(device)
          for prediction in prediction_perm:
              ## Old evaluation measure [-pi/2, pi/2] 
              error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2                        # Calculate error with modulo pi
              rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)                          # Calculate MSE 
              rmspe_list.append(rmspe_val)
          rmspe_tensor = torch.stack(rmspe_list, dim = 0)
          rmspe_min = torch.min(rmspe_tensor)
          rmspe.append(rmspe_min)
      result = torch.sum(torch.stack(rmspe, dim = 0))
      return result
class MSPELoss(nn.Module):
    def __init__(self):
        super(MSPELoss, self).__init__()
    def forward(self, preds, DOA):
        rmspe = []
        for iter in range(preds.shape[0]):
            rmspe_list = []
            Batch_preds = preds[iter].to(device)
            targets = DOA[iter].to(device)
            prediction_perm = permute_prediction(Batch_preds).to(device)
            for prediction in prediction_perm:
                ## Old evaluation measure [-pi/2, pi/2] 
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2                        # Calculate error with modulo pi
                rmspe_val = (1 / len(targets)) * (torch.linalg.norm(error) ** 2)                           # Calculate MSE 
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            rmspe_min = torch.min(rmspe_tensor)
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result

def RMSPE(pred, DOA):
    rmspe_list = []
    for p in list(permutations(pred, len(pred))):
        p = np.array(p)
        DOA = np.array(DOA)
        error = (((p - DOA) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        rmspe_val = (1 / np.sqrt(len(p))) * np.linalg.norm(error)
        rmspe_list.append(rmspe_val)
    return np.min(rmspe_list)

def MSPE(pred, DOA):
    rmspe_list = []
    for p in list(permutations(pred, len(pred))):
        p = np.array(p)
        DOA = np.array(DOA)
        error = (((p - DOA) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        rmspe_val = (1 / len(p)) * (np.linalg.norm(error) ** 2)
        rmspe_list.append(rmspe_val)
    return np.min(rmspe_list)