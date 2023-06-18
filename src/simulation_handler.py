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
import warnings
import time
import copy
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from src.methods import *
from src.models import *
from src.criterions import *
from src.utils import *

# Initialization
warnings.simplefilter("ignore")
plt.close('all')
set_unified_seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Saving path
saving_path = r"C:\Users\dorsh\Deep RootMUSIC\Code\Weights\Models"

def run_simulation(train_dataset: tuple, test_dataset: tuple,
                    tau: int, optimizer_name: str, lr_val: float, schedular: bool,
                    weight_decay_val: float, step_size_val: float, gamma_val: float,
                    num_epochs: int, model_name: str, batch_size: int, system_model: SystemModel,
                    load_flag:bool = False, loading_path:str = None, Plot:bool = True,
                    saving_path:str = saving_path, activation_value = 0.5, model_type = "SubNet"):
  
    # Set the seed for all available random operations
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)

    ## Model initialization
    # Assign the desired model
    if model_type.startswith("DA-MUSIC"):
      model = DeepAugmentedMUSIC(N=system_model.N, T=system_model.T, M=system_model.M)
    elif model_type.startswith("DeepCNN"):
      model = DeepCNN(N=system_model.N, grid_size=361)
    else:
      model = SubspaceNet(tau=tau, M=system_model.M)
    # Load model to specified device, either gpu or cpu
    model = model.to(device)                                   
    # Load model weights
    if load_flag == True:
      if torch.cuda.is_available() == False:
        model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
        print("CPU")
      else:
        model.load_state_dict(torch.load(loading_path))
      print("Loaded Successfully")
    ## Training parameters 
    # Assign optimizer 
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr_val,weight_decay=weight_decay_val)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr_val)
    elif optimizer_name == "SGD Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr_val, momentum=0.9)
    # Assign schedular for learing rate decay
    if schedular:
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=step_size_val, gamma=gamma_val)
    # Define loss criterion
    if model_type.startswith("DeepCNN"):
      criterion = nn.BCELoss() 
    else:
      criterion = RMSPELoss()

    ## Generate training and testing dataloader objects
    # Divide into training and validation datasets 
    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.1, shuffle = True)
    print("Training DataSet size", len(train_dataset))
    print("Validation DataSet size", len(valid_dataset))
    # Transform datasets into DataLoader objects    
    train_dataset = torch.utils.data.DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, drop_last=False)  
    valid_dataSet = torch.utils.data.DataLoader(valid_dataset,
            batch_size=1, shuffle=False, drop_last=False)
    # Transform test-dataset into DataLoader object
    print("Test_DataSet", len(test_dataset))
    test_dataset = torch.utils.data.DataLoader(test_dataset,
            batch_size=1, shuffle=False, drop_last=False)
    ## Train model using the train_model function
    model, loss_train_list, loss_valid_list = train_model(model, train_dataset, valid_dataSet,
                 optimizer, criterion, epochs= num_epochs, model_name=model_name, scheduler=lr_decay,
                    checkpoint_path=r"C:\Users\dorsh\Deep RootMUSIC\Code\Weights" + '\ckpt-{}.pk', model_type=model_type)
    # Save models best weights
    torch.save(model.state_dict(), saving_path + '\\model_' + dt_string_for_save)
    # Plot learning and validation loss curves
    if Plot:
      plot_learning_curve(list(range(num_epochs)), loss_train_list, loss_valid_list)
    ## Evaluation stage
    print("\n--- Evaluating Stage ---\n")
    if model_type.startswith("DeepCNN"):
      model_loss = -1
    else:
      # Compute the model overall loss
      model_loss = evaluate_model(model, test_dataset, criterion, model_type=model_type)
      print("{} Test loss = {}".format(model_type, model_loss))
    return model, loss_train_list, loss_valid_list, model_loss

def train_model(model, Train_data, Valid_data,
                 optimizer, criterion, epochs,
                 model_name, scheduler=None, checkpoint_path=None, model_type="SubNet"):
    PRINT_WEIGHTS = False
    since = time.time()
    loss_train_list = []
    loss_valid_list = []
    min_valid_loss = np.inf
    print("\n---Start Training Stage ---\n")

    for epoch in range(epochs):
        ## Train the model for a specific epoch
        train_length = 0
        model.train()
        Overall_train_loss = 0.0
        model = model.to(device)
        
        for i, data in enumerate(tqdm(Train_data)):
            Train_data
            Rx, DOA = data
            # print("DOA", DOA * R2D)
            train_length += DOA.shape[0]
            Rx = Variable(Rx, requires_grad=True).to(device)
            DOA = Variable(DOA, requires_grad=True).to(device)

            model_parameters = model(Rx)
            if model_type.startswith("DA-MUSIC") or model_type.startswith("DeepCNN"):
              # Deep Augmented MUSIC or DeepCNN
              DOA_predictions = model_parameters
            else:
              # Default - SubSpaceNet
              DOA_predictions = model_parameters[0]

            ## Compute training loss
            if model_type.startswith("DeepCNN"):
              train_loss = criterion(DOA_predictions.float(), DOA.float())
            else:
              train_loss = criterion(DOA_predictions, DOA)

            ## Back-propagation stage
            try:                         
              train_loss.backward()
            except RuntimeError:
              print("linalg error")
              pass

            ## perform parameter update
            optimizer.step()                                                     
            
            model.zero_grad()                                                   # reset the gradients back to zero
            if model_type.startswith("DeepCNN") or model_type.startswith("DA-MUSIC"):
              Overall_train_loss += train_loss.item() * len(data[0])                             # add the batch training loss to epoch loss
            else:
              Overall_train_loss += train_loss.item()                            # add the batch training loss to epoch loss

            # print("iteration loss : ",train_loss.item())
            # if i % 100 == 0:
            #   print("Iteration = {}, accumulated loss= {}".format(i+1, Overall_train_loss / (i+1)))
            
            if PRINT_WEIGHTS:
                for name, param in model.named_parameters():
                  if param.grad is not None:
                    print(name, param.grad.sum())
                else:
                    print(name, param.grad)
        # print("len(Train_data)", len(Train_data))
        # print("Overall_train_loss = {}, train_length = {}".format(Overall_train_loss, train_length))
        Overall_train_loss = Overall_train_loss / train_length # compute the epoch training loss
        loss_train_list.append(Overall_train_loss)
        if scheduler != None:
            scheduler.step()
        ## Evaluate the model for a specific epoch
        Overall_valid_loss = 0.0
        model.eval()
        valid_length = 0
        
        with torch.no_grad():                                                   # Gradients calculation isn't required for evaluation
            for i, data in enumerate(Valid_data):
                Rx, DOA = data
                valid_length += DOA.shape[0]
                Rx = Rx.to(device)
                DOA = DOA.to(device)
                model_parameters = model(Rx)
                if model_type.startswith("DA-MUSIC") or model_type.startswith("DeepCNN"):
                  # Deep Augmented MUSIC or DeepCNN
                  DOA_predictions = model_parameters
                else:
                  # Default - SubSpaceNet
                  DOA_predictions = model_parameters[0]
                if model_type.startswith("DeepCNN"):
                  eval_loss = criterion(DOA_predictions.float(), DOA.float())
                else:
                  eval_loss = criterion(DOA_predictions, DOA)                 # Compute evaluation predictions loss
                Overall_valid_loss += eval_loss.item()                          # add the batch evaluation loss to epoch loss
            Overall_valid_loss = Overall_valid_loss / valid_length
            loss_valid_list.append(Overall_valid_loss)
        
        ## Report results
        print("epoch : {}/{}, Train loss = {:.6f}, Validation loss = {:.6f}".format(epoch + 1,
                         epochs, Overall_train_loss, Overall_valid_loss))                       # display the epoch training loss
        print('lr {}'.format(optimizer.param_groups[0]['lr']))
        
        ## save model weights for better validation performances
        if min_valid_loss > Overall_valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{Overall_valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = Overall_valid_loss
            best_epoch = epoch
            ## Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), saving_path + '\\' +  model_name)
    
    time_elapsed = time.time() - since
    # plot_learning_curve(list(range(epochs)),
    #                     loss_train_list, loss_valid_list)
    print("\n--- Training summary ---")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Minimal Validation loss: {:4f} at epoch {}'.format(min_valid_loss, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), saving_path + '//' +  model_name)
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

def evaluate_model(model, Data, criterion, plot_spec = False, figures = None, model_type="SubNet"):
    loss = 0.0
    model.eval()
    test_length = 0
    with torch.no_grad():                                                                   # Gradients Calculation isnt required for evaluation
        for i, data in enumerate(Data):
            Rx, DOA = data
            test_length += DOA.shape[0]
            Rx = Rx.to(device)
            DOA = DOA.to(device)
            model_parameters = model(Rx)
            if model_type.startswith("DA-MUSIC"):
              # Deep Augmented MUSIC
              DOA_predictions = model_parameters
            elif model_type.startswith("DeepCNN"):
              # CNN
              DOA_predictions = model_parameters
              DOA_predictions = get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
              DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
            else:
              # Default - SubSpaceNet
              DOA_predictions = model_parameters[0]
            eval_loss = criterion(DOA_predictions, DOA)                                     # Compute evaluation predictions loss
            loss += eval_loss.item()                                          # add the batch evaluation loss to epoch loss  
        loss = loss / test_length
    if plot_spec:
      DOA_all = model_parameters[1]
      roots = model_parameters[2]
      plot_spectrum(DOA_prediction=DOA_all * R2D, true_DOA=DOA[0] * R2D, roots=roots,
                    algorithm="SubNet+R-MUSIC", figures=figures)
    return loss

def evaluate_hybrid_model(model_hybrid, Data, system_model, criterion = RMSPE,
    model_name=None, algorithm = "music", plot_spec = False, figures = None):
  # Initialize parameters for evaluation
  mb_methods = ModelBasedMethods(system_model)
  hybrid_loss = [] 
  model_hybrid.eval()
  # Gradients calculation isn't required for evaluation
  with torch.no_grad():   
    for i, data in enumerate(Data):
      Rx, DOA = data
      Rx = Rx.to(device)
      DOA = DOA.to(device)
            
      ## Hybrid MUSIC
      if algorithm.startswith("music"):
        DOA_pred, spectrum, M = mb_methods.hybrid_MUSIC(model_hybrid, Rx, system_model.scenario)
        DOA_pred = mb_methods.angels[DOA_pred] * R2D
        # Take the first M predictions
        predicted_DOA = DOA_pred[:M][::-1]  
        while(predicted_DOA.shape[0] < M):
          print("Cant estimate M sources - hybrid {}".format(algorithm))
          predicted_DOA = np.insert(predicted_DOA, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)   
        # Calculate loss criterion
        loss = criterion(predicted_DOA, DOA * R2D)
        hybrid_loss.append(loss)
        if plot_spec and i == len(Data.dataset) - 1:
          figures["music"]["norm factor"] = np.max(spectrum)
          plot_spectrum(DOA_prediction=predicted_DOA, true_DOA=DOA * R2D, system_model=system_model,
                        spectrum=spectrum, algorithm="SubNet+MUSIC", figures=figures)
        
      ## Hybrid ESPRIT
      elif algorithm.startswith("esprit"):
        predicted_DOA, M = mb_methods.esprit(X=None, HYBRID = True, model_ESPRIT=model_hybrid, Rz=Rx, scenario=system_model.scenario)
        while(predicted_DOA.shape[0] < M):
          print("Cant estimate M sources - hybrid {}".format(algorithm))
          predicted_DOA = np.insert(predicted_DOA, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)   
        loss = criterion(predicted_DOA, DOA * R2D)
        hybrid_loss.append(loss)
        # if plot_spec and i == len(Data.dataset) - 1: 
          # plot_spectrum(DOA_prediction=predicted_DOA, true_DOA=DOA * R2D, system_model=system_model, spectrum=spectrum, algorithm=algorithm)
            
      ## Hybrid MVDR
      elif algorithm.startswith("mvdr"):
        # mb.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 3600, endpoint=False)
        MVDR_spectrum = mb.MVDR(X=None, HYBRID = True, model_mvdr=model_hybrid, Rz=Rx, scenario=system_model.scenario)
        if plot_spec and i == len(Data.dataset) - 1:
          figures["mvdr"]["norm factor"] = np.max(MVDR_spectrum)
          plot_spectrum(DOA_prediction=None, true_DOA=DOA * R2D, system_model=system_model, spectrum=MVDR_spectrum, algorithm="SubNet+MVDR", figures=figures)
        hybrid_loss = 0
      
      else:
        return None
  return np.mean(hybrid_loss)


def evaluate_model_based(dataset_mb, system_model, criterion=RMSPE, plot_spec=False, algorithm="music", figures=None):
  loss_list = []
  mb = ModelBasedMethods(system_model)
  for i, data in enumerate(dataset_mb):
    X, doa = data
    X = X[0]
    # TODO: unify BB music and music under the same method
    ### Root-MUSIC algorithms ###
    if "r-music" in algorithm:
      if algorithm.startswith("sps"):
        DOA_pred, roots, M, DOA_pred_all, _ = mb.root_music(X, NUM_OF_SOURCES=True,
                SPS=True, sub_array_size=int(mb.N / 2) + 1)
      else:
        DOA_pred, roots, M, DOA_pred_all, _ = mb.root_music(X, NUM_OF_SOURCES=True)
      # if algorithm cant estimate M sources, randomize angels
      while(DOA_pred.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        DOA_pred = np.insert(DOA_pred, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)        
      loss = criterion(DOA_pred, doa * R2D)
      loss_list.append(loss)
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=DOA_pred_all, true_DOA=doa[0] * R2D, roots=roots,
                      algorithm=algorithm.upper(), figures= figures)
    ### MUSIC algorithms ###
    elif "music" in algorithm:
      if algorithm.startswith("bb"):
        DOA_pred, MUSIC_Spectrum, M = mb.broadband_MUSIC(X)
      elif algorithm.startswith("sps"):
        DOA_pred, MUSIC_Spectrum, M = mb.MUSIC(X, NUM_OF_SOURCES=True,
                                      SPS=True, sub_array_size=int(mb.N / 2) + 1)
      elif algorithm.startswith("music"):
            DOA_pred, MUSIC_Spectrum, M = mb.MUSIC(X, scenario=system_model.scenario)
      
      DOA_pred = mb.angels[DOA_pred] * R2D  # Convert from radians to degrees
      predicted_DOA = DOA_pred[:M][::-1]  # Take First M predictions
      # if algorithm cant estimate M sources, randomize angels
      while(predicted_DOA.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        predicted_DOA = np.insert(predicted_DOA, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)
      loss = criterion(predicted_DOA, doa * R2D)
      loss_list.append(loss)
      # plot BB-MUSIC spectrum
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=predicted_DOA, true_DOA=doa * R2D,
              system_model=system_model, spectrum=MUSIC_Spectrum, algorithm=algorithm.upper(), figures=figures)
    
    ### ESPRIT algorithms ###
    elif "esprit" in algorithm:
      if algorithm.startswith("sps"):
        DOA_pred, M = mb.esprit(X, NUM_OF_SOURCES=True, SPS=True, sub_array_size=int(mb.N / 2) + 1)
      else:
        DOA_pred, M = mb.esprit(X, NUM_OF_SOURCES=True, scenario=system_model.scenario)
      # if algorithm cant estimate M sources, randomize angels
      while(DOA_pred.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        DOA_pred = np.insert(DOA_pred, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)        
      loss = criterion(DOA_pred, doa * R2D)
      loss_list.append(loss)

    # MVDR evaluation
    elif algorithm.startswith("mvdr"):
      MVDR_spectrum = mb.MVDR(X=X, scenario=system_model.scenario)
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=None, true_DOA=doa * R2D, system_model=system_model,
                      spectrum=MVDR_spectrum, algorithm=algorithm.upper(), figures= figures)
  return np.mean(loss_list)


def plot_spectrum(DOA_prediction, true_DOA, system_model=None, spectrum=None, roots=None,
                  algorithm="music", figures = None):
  if isinstance(DOA_prediction, (np.ndarray, list, torch.Tensor)):
    DOA_prediction = np.squeeze(np.array(DOA_prediction))
  # MUSIC algorithms
  if "music" in algorithm.lower() and not ("r-music" in algorithm.lower()):
    if figures["music"]["fig"] == None:
      plt.style.use('default')
      figures["music"]["fig"] = plt.figure(figsize=(8, 6))
      plt.style.use('plot_style.txt')
    if figures["music"]["ax"] == None:
      figures["music"]["ax"] = figures["music"]["fig"].add_subplot(111)

    mb = ModelBasedMethods(system_model)
    angels_grid = mb.angels * R2D
    # ax.set_title(algorithm.upper() + "spectrum")
    figures["music"]["ax"].set_xlabel("Angels [deg]")
    figures["music"]["ax"].set_ylabel("Amplitude")
    figures["music"]["ax"].set_ylim([0.0, 1.01])
    figures["music"]["norm factor"] = None
    if figures["music"]["norm factor"] != None:
      figures["music"]["ax"].plot(angels_grid , spectrum / figures["music"]["norm factor"], label=algorithm)
    else:
      figures["music"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    figures["music"]["ax"].legend()

  elif "mvdr" in algorithm.lower():
    if figures["mvdr"]["fig"] == None:
      plt.style.use('default')
      figures["mvdr"]["fig"] = plt.figure(figsize=(8, 6))
      plt.style.use('plot_style.txt')
    if figures["mvdr"]["ax"] == None:
      figures["mvdr"]["ax"] = figures["mvdr"]["fig"].add_subplot(111, polar=True)
    mb = ModelBasedMethods(system_model)
    # mb.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 3600, endpoint=False)
    # ax.set_xlabel("Angels [deg]")
    figures["mvdr"]["ax"].set_theta_zero_location('N')
    figures["mvdr"]["ax"].set_theta_direction(-1)
    figures["mvdr"]["ax"].set_thetamin(-90)
    figures["mvdr"]["ax"].set_thetamax(90)
    
    angels_grid = mb.angels
    # figures["mvdr"]["ax"].set_title(algorithm.upper() + "spectrum")
    # ax.set_xlabel("Angels [deg]")
    # ax.set_ylabel("Amplitude")
    figures["mvdr"]["ax"].set_ylim([0.0, 1.01])
    figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    for doa in true_DOA[0]:
      figures["mvdr"]["ax"].plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm + " pattern")
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum),  label=algorithm.upper() + " pattern")
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum, label=algorithm + " pattern")
    figures["mvdr"]["ax"].legend()
  
  elif "r-music" in algorithm.lower():
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    plt.style.use('plot_style.txt')
    ax = fig.add_subplot(111, polar=True)

    # ax.set_xlabel("Angels [deg]")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)

    for i in range(len(DOA_prediction)):
      angle = DOA_prediction[i]
      r = np.abs(roots[i])
      ax.set_ylim([0, 1.2])
      # ax.set_yticks([0, 1, 2, 3, 4])
      ax.set_yticks([0, 1])
      ax.plot([0, angle * np.pi / 180], [0, r], marker='o')
    for doa in true_DOA:
      ax.plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    # ax.set_xlabel("Angels [deg]")
    # ax.set_ylabel("Amplitude")
    plt.savefig("{}_spectrum.pdf".format(algorithm), bbox_inches='tight')
