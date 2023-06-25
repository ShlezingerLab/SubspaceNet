"""Subspace-Net
Details
----------
Name: simulation_handler.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This script defines some helpful functions:
    * run_simulation: 
    * train_model:
    * plot_learning_curve:
    * evaluate_model:
    * evaluate_model_based:
    * plot_spectrum:
    * evaluate_model_based:
"""

# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import copy
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from src.utils import *
from src.criterions import *
from src.system_model import SystemModel
from src.methods import MUSIC, RootMUSIC, MVDR, Esprit
from src.models import SubspaceNet, DeepCNN, DeepAugmentedMUSIC
from src.evaluation import evaluate_model

class TrainingParams(object):
  def __init__(self, model_type: str) -> None:
    self.model_type = model_type
    
  def set_batch_size(self, batch_size: int):
    self.batch_size = batch_size
    return self

  def set_epochs(self, epochs:int):
    self.epochs = epochs
    return self
  
  def set_model(self, system_model: SystemModel, tau:int = None):
    # Assign the desired model for training
    if self.model_type.startswith("DA-MUSIC"):
      model = DeepAugmentedMUSIC(N=system_model.N, T=system_model.T, M=system_model.M)
    elif self.model_type.startswith("DeepCNN"):
      model = DeepCNN(N=system_model.N, grid_size=361)
    elif self.model_type.startswith("SubspaceNet"):
      model = SubspaceNet(tau=tau, M=system_model.M)
    else:
      raise Exception(f"Train.set_model: Model type {self.model_type} is not defined")
    # assign model to device
    self.model = model.to(device)
    return self
  
  def load_model(self, load_flag: bool, loading_path: Path):
    # Load model from given path
    if load_flag:
      self.model.load_state_dict(torch.load(loading_path, map_location=device))
    return self
  
  def set_optimizer(self, optimizer: str, learning_rate: float, weight_decay: float):
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    # Assign optimizer for training
    if optimizer.startswith("Adam"):
      self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,\
        weight_decay=weight_decay)
    elif optimizer.startswith("SGD"):
      self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    elif optimizer == "SGD Momentum":
      self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,\
        momentum=0.9)
    else:
      raise Exception(f"Train.set_optimizer: Optimizer {optimizer} is not defined")
    return self
  
  def set_schedular(self, step_size: float, gamma: float):
    # Number of steps for learning rate decay iteration
    self.step_size = step_size
    # learning rate decay value
    self.gamma = gamma
    # Assign schedular for learning rate decay
    self.schedular = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    return self
  
  def set_criterion(self):
    # Define loss criterion
    if self.model_type.startswith("DeepCNN"):
      self.criterion = nn.BCELoss() 
    else:
      self.criterion = RMSPELoss()
    return self
  
  def set_training_dataset(self, train_dataset:list):
      # Divide into training and validation datasets
      train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.1, shuffle = True)
      print("Training DataSet size", len(train_dataset))
      print("Validation DataSet size", len(valid_dataset))
      # Transform datasets into DataLoader objects    
      self.train_dataset = torch.utils.data.DataLoader(train_dataset,
          batch_size=self.batch_size, shuffle=True, drop_last=False)  
      self.valid_dataset = torch.utils.data.DataLoader(valid_dataset,
          batch_size=1, shuffle=False, drop_last=False)
      return self



def train(training_parameters:TrainingParams, model_name: str,
    plot_curves:bool = True, saving_path:Path = None):
    # Set the seed for all available random operations
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model 
    model, loss_train_list, loss_valid_list = train_model(training_parameters, model_name=model_name, checkpoint_path= saving_path)
    # Save models best weights
    torch.save(model.state_dict(), saving_path / Path(dt_string_for_save))
    # Plot learning and validation loss curves
    if plot_curves:
      plot_learning_curve(list(range(training_parameters.epochs)), loss_train_list, loss_valid_list)
    return model, loss_train_list, loss_valid_list

def train_model(training_params: TrainingParams, model_name, checkpoint_path=None):
    model = copy.deepcopy(training_params.model)
    since = time.time()
    loss_train_list = []
    loss_valid_list = []
    min_valid_loss = np.inf
    print("\n---Start Training Stage ---\n")
    # Run over all epochs
    for epoch in range(training_params.epochs):
        train_length = 0
        overall_train_loss = 0.0
        # Set model to train mode
        model.train()
        model = model.to(device)
        for data in tqdm(training_params.train_dataset):
            Rx, DOA = data
            train_length += DOA.shape[0]
            # Cast observations and DoA to Variables
            Rx = Variable(Rx, requires_grad=True).to(device)
            DOA = Variable(DOA, requires_grad=True).to(device)
            # Get model output
            model_output = model(Rx)
            if training_params.model_type.startswith("SubspaceNet"):
              # Default - SubSpaceNet
              DOA_predictions = model_output[0]
            else:
              # Deep Augmented MUSIC or DeepCNN
              DOA_predictions = model_output
            # Compute training loss
            if training_params.model_type.startswith("DeepCNN"):
              train_loss = training_params.criterion(DOA_predictions.float(), DOA.float())
            else:
              train_loss = training_params.criterion(DOA_predictions, DOA)
            # Back-propagation stage
            try:
              train_loss.backward()
            except RuntimeError:
              print("linalg error")
            # optimizer update
            training_params.optimizer.step()                                                     
            # reset gradients
            model.zero_grad()
            # add batch loss to overall epoch loss
            if training_params.model_type.startswith("DeepCNN"):
              # BCE is averaged
              overall_train_loss += train_loss.item() * len(data[0])
            else:
              # RMSPE is summed
              overall_train_loss += train_loss.item()
        # Average the epoch training loss
        overall_train_loss = overall_train_loss / train_length
        loss_train_list.append(overall_train_loss)
        # Update schedular
        training_params.schedular.step()
        # Calculate evaluation loss
        valid_loss = evaluate_model(model, training_params.train_dataset,
            training_params.criterion, model_type=training_params.model_type)
        loss_valid_list.append(valid_loss)
        # Report results
        print("epoch : {}/{}, Train loss = {:.6f}, Validation loss = {:.6f}".format(epoch + 1,\
            training_params.epochs, overall_train_loss, valid_loss))
        print('lr {}'.format(training_params.optimizer.param_groups[0]['lr']))
        # save best model weights for early stooping
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            best_epoch = epoch
            # Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path / model_name)
    
    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimal Validation loss: {:4f} at epoch {}'.format(min_valid_loss, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path /  model_name)
    return model, loss_train_list, loss_valid_list

def plot_learning_curve(epoch_list, train_loss, Validation_loss):
    """
    Plot the learning curve.
    """
    plt.title("Learning Curve: Loss per Epoch")
    plt.plot(epoch_list, train_loss, label="Train")
    plt.plot(epoch_list, Validation_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def simulation_summary(model_type: str, M: int, N: int, T: float, SNR: int,\
                scenario: str, mode: str, eta: float, geo_noise_var: float,\
                training_parameters:TrainingParams = None, phase = "training", tau: int = None):
    """
    Prints a summary of the simulation parameters.

    Args:
    -----
        model_type (str): The type of the model.
        M (int): The number of sources.
        N (int): The number of sensors.
        T (float): The number of observations.
        SNR (int): The signal-to-noise ratio.
        scenario (str): The scenario of the signals.
        mode (str): The nature of the sources.
        eta (float): The spacing deviation.
        geo_noise_var (float): The geometry noise variance.
        training_parameters (TrainingParams): instance of the training parameters object
        phase (str, optional): The phase of the simulation. Defaults to "training", optional: "evaluation".
        tau (int, optional): The number of lags for auto-correlation (relevant only for SubspaceNet model).

    """
    print("\n--- New Simulation ---\n")
    print(f"Description: Simulation of {model_type}, {phase} stage")
    print("System model parameters:")
    print(f"Number of sources = {M}")
    print(f"Number of sensors = {N}")
    print(f"scenario = {scenario}")
    print(f"Observations = {T}")
    print(f"SNR = {SNR}, {mode} sources")
    print(f"Spacing deviation (eta) = {eta}")
    print(f"Geometry noise variance = {geo_noise_var}")
    print("Simulation parameters:")
    print(f"Model: {model_type}")
    if model_type.startswith("SubspaceNet"):
        print("Tau = {}".format(tau))
    if phase.startswith("training"):
        print(f"Epochs = {training_parameters.epochs}")
        print(f"Batch Size = {training_parameters.batch_size}")
        print(f"Learning Rate = {training_parameters.learning_rate}")
        print(f"Weight decay = {training_parameters.weight_decay}")
        print(f"Gamma Value = {training_parameters.gamma}")
        print(f"Step Value = {training_parameters.step_size}")