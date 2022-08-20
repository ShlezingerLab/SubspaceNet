import torch
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import warnings
import time
import copy
import torch.optim as optim
from datetime import datetime
from itertools import permutations
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from DataLoaderCreation import *
from Signal_creation import *
from methods import *
from models import *
from EvaluationMesures import *

warnings.simplefilter("ignore")
plt.close('all')

def Set_Overall_Seed(SeedNumber = 42):
  random.seed(SeedNumber)
  np.random.seed(SeedNumber)
  torch.manual_seed(SeedNumber)

Set_Overall_Seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saving_path = r"G:\My Drive\Thesis\DeepRootMUSIC\Code\Weights\Models"

def Run_Simulation(Model_Train_DataSet,
                    Model_Test_DataSet,
                    tau, N, optimizer_name, lr_val, Schedular,
                    weight_decay_val, step_size_val, gamma_val, num_epochs,
                    model_name,
                    Bsize,
                    Sys_Model,
                    ActivationVal = 0.5,
                    checkpoint_optimizer_path = None,
                    load_flag = False, loading_path = None,
                    Plot = True, DataSetModelBased = None,
                    Plot_Spectrum_flag = False,
                    saving_path = saving_path):
  
    ## Set the seed for all available random operations
    Set_Overall_Seed()
    
    ## Current date and time
    print("\n----------------------\n")

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)

    ############################
    ###    Compare Models    ###
    ############################

    ## Transform model-based test dataset into DataLoader Object:
    if DataSetModelBased != None:
      print("Test_DataSet", len(Model_Test_DataSet))
      DataSetModelBased = torch.utils.data.DataLoader(DataSetModelBased,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)
      
    ## Compute MUSIC and Root-MUSIC algorithms overall loss:
      RootMUSIC_loss, MUSIC_loss = evaluate_model_based(DataSetModelBased, Sys_Model)
      print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))
      print("MUSIC Test loss = {}".format(MUSIC_loss))

    ############################
    ### Model initialization ###
    ############################

    # Create a model from `Deep_Root_Net`
    # model = Deep_Root_Net(tau=tau, ActivationVal=ActivationVal)                              
    model = Deep_Root_Net_AntiRectifier(tau=tau, ActivationVal=ActivationVal)                              
    # model = Deep_Root_Net_AntiRectifier_Extend(tau=tau, ActivationVal=ActivationVal)                              
    
    # Load it to the specified device, either gpu or cpu
    model = model.to(device)                                   
    
    ## Loading available model
    if load_flag == True:
      if torch.cuda.is_available() == False:
        model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
        print("CPU")
      else:
        model.load_state_dict(torch.load(loading_path))
      print("Loaded Succesfully")
    
    ## Create an optimizer 
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr_val,weight_decay=weight_decay_val)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr_val)
    elif optimizer_name == "SGD Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr_val, momentum=0.9)
    if Schedular:
        lr_decay = lr_scheduler.StepLR(optimizer, step_size=step_size_val, gamma=gamma_val)

    ## Loss criterion
    criterion = PRMSELoss()                                     # Periodic rmse loss

    ############################
    ###  Data Organization   ###
    ############################

    ## Split data into Train and Validation
    Train_DataSet, Valid_DataSet = train_test_split(Model_Train_DataSet, test_size=0.1, shuffle = True)
    print("Training DataSet size", len(Train_DataSet))
    print("Validation DataSet size", len(Valid_DataSet))

    ## Transform Training Datasets into DataLoader Object    
    Train_data = torch.utils.data.DataLoader(Train_DataSet,
                                    batch_size=Bsize,
                                    shuffle=True,
                                    drop_last=False)  
    Valid_data = torch.utils.data.DataLoader(Valid_DataSet,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
    
    ## Transform Test Dataset into DataLoader Object
    print("Test_DataSet", len(Model_Test_DataSet))
    Test_data = torch.utils.data.DataLoader(Model_Test_DataSet,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
    
    ############################
    ###     Train Model      ###
    ############################

    ## Train using the "train_model" function
    model, loss_train_list, loss_valid_list = train_model(model, Train_data, Valid_data,
                 optimizer, criterion, epochs= num_epochs, model_name=model_name, scheduler=lr_decay,
                    checkpoint_path=r"G:\My Drive\Thesis\\DeepRootMUSIC\Code\\Weights" + '\ckpt-{}.pk')
    
    ## Save model Best weights
    torch.save(model.state_dict(), saving_path + '\\' +  model_name + dt_string_for_save)
    
    ############################
    ###    Evaluate Model    ###
    ############################
    print("\n--- Evaluating Stage ---\n")
    ## Plot learning and validation loss curves
    if Plot:
      plot_learning_curve(list(range(num_epochs)), loss_train_list, loss_valid_list)

    ## Compute the model Overall loss
    DeepRootTest_loss = evaluate_model(model, Test_data, criterion)
    print("Deep Root-MUSIC Test loss = {}".format(DeepRootTest_loss))
    print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))
    print("MUSIC Test loss = {}".format(MUSIC_loss))

    ############################
    ###   Model's spectrum   ###
    ############################
    if Plot_Spectrum_flag:
      PlotSpectrum(model)
    
    return model, loss_train_list, loss_valid_list, DeepRootTest_loss


def train_model(model, Train_data, Valid_data,
                 optimizer, criterion, epochs,
                 model_name, scheduler=None, checkpoint_path=None):
    PRINT_WEIGHTS = False
    since = time.time()
    loss_train_list = []
    loss_valid_list = []
    min_valid_loss = np.inf
    print("\n---Start Training Stage ---\n")

    for epoch in tqdm(range(epochs)):
        ## Train the model for a specific epoch
        train_length = 0
        model.train()
        Overall_train_loss = 0.0
        model = model.to(device)
        
        for i, data in enumerate(Train_data):
            Rx, DOA = data
            # print("DOA", DOA * 180 / np.pi)
            train_length += DOA.shape[0]
            Rx = Variable(Rx, requires_grad=True).to(device)
            DOA = Variable(DOA, requires_grad=True).to(device)
            
            ## Compute model DOA predictions  
            model_parameters = model(Rx, DOA.shape[1])
                                        
            # DOA_predictions = model_parameters
            DOA_predictions = model_parameters[0]

            ## Compute training loss
            train_loss = criterion(DOA_predictions, DOA)

            ## Backpropogation stage
            try:                         
              train_loss.backward()
            except RuntimeError:
              print("linalg error")
              pass

            ## perform parameter update
            optimizer.step()                                                     
            
            model.zero_grad()                                                   # reset the gradients back to zero
            Overall_train_loss += train_loss.item()                             # add the batch training loss to epoch loss

            # print("iteration loss : ",train_loss.item())
            # if i % 50 == 0:
            #   print("Iteration = {}, accumulated loss= {}".format(i+1, Overall_train_loss / (i+1)))
            
            if PRINT_WEIGHTS:
              for name, param in model.named_parameters():
                if param.grad is not None:
                  print(name, param.grad.sum())
                else:
                  print(name, param.grad)
        # print("len(Train_data)", len(Train_data))
        # print("Overall_train_loss = {}, train_length = {}".format(Overall_train_loss, train_length))
        Overall_train_loss = Overall_train_loss / train_length               # compute the epoch training loss
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
                model_parameters = model(Rx, DOA.shape[1])                            # Compute prediction of DOA's
                DOA_predictions = model_parameters[0]
                # DOA_predictions = model_parameters
                eval_loss = criterion(DOA_predictions, DOA)                     # Compute evaluation predictions loss
                Overall_valid_loss += eval_loss.item()                          # add the batch evaluation loss to epoch loss
            
            Overall_valid_loss = Overall_valid_loss / valid_length
            loss_valid_list.append(Overall_valid_loss)
        
        ## Report results
        print("epoch : {}/{}, Train loss = {:.6f}, Validation loss = {:.6f}".format(epoch + 1,
                         epochs, Overall_train_loss, Overall_valid_loss))                       # display the epoch training loss
        print('lr {}'.format(optimizer.param_groups[0]['lr']))
        
        ## save model weights for better validation performences
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
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
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

def evaluate_model(model, Data, criterion):
    loss = 0.0
    model.eval()
    test_length = 0
    with torch.no_grad():                                                                   # Gradients Calculation isnt required for evaluation
        for i, data in enumerate(Data):
            Rx, DOA = data
            test_length += DOA.shape[0]
            Rx = Rx.to(device)
            DOA = DOA.to(device)
            model_parameters = model(Rx, DOA.shape[1])                            # Compute prediction of DOA's
            # DOA_predictions = model_parameters
            DOA_predictions = model_parameters[0]
            eval_loss = criterion(DOA_predictions, DOA)                                     # Compute evaluation predictions loss
            loss += eval_loss.item()                                          # add the batch evaluation loss to epoch loss
        # print(len(Data))    
        loss = loss / test_length
    return loss


def PRMSE(pred, DOA):
  prmse_list = []
  for p in list(permutations(pred, len(pred))):
      p = np.array(p)
      DOA = np.array(DOA)
      error = (((p - DOA) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
      # error = ((((p - DOA) * np.pi / 180) + np.pi) % (2 * np.pi)) - np.pi 
      # error = ((((p - DOA) * np.pi / 180)) % (2 * np.pi))
      # error = ((((p - DOA) * np.pi / 180)) % (2 * np.pi))
      prmse_val = (1 / np.sqrt(len(p))) * np.linalg.norm(error)
      prmse_list.append(prmse_val)
  return np.min(prmse_list)

def evaluate_model_based(DataSetModelBased, Sys_Model):
  RootMUSIC_list = []
  MUSIC_list = []
  DeepRootMUSIC_list = []
  model_based_platform = Model_Based_methods(Sys_Model)
  for i,data in enumerate(DataSetModelBased):
      X, Y = data
      X_modelbased = X[0]
      ## RootMUSIC predictions
      DOA_pred_RootMUSIC, roots, M, DOA_pred_all, roots_angels_all = model_based_platform.Classic_Root_MUSIC(X_modelbased, NUM_OF_SOURCES=True)
      
      if(DOA_pred_RootMUSIC.shape[0] < M):
        print("Cant estimate M sources - RootMUSIC")
              
      else:
        lossRootMUSIC = PRMSE(DOA_pred_RootMUSIC, Y * 180 / np.pi)
        RootMUSIC_list.append(lossRootMUSIC)
      
      ## MUSIC predictions
      DOA_pred_MUSIC, Spectrum, M = model_based_platform.Classic_MUSIC(X_modelbased, NUM_OF_SOURCES=M)
      DOA_pred = model_based_platform.angels[DOA_pred_MUSIC] * 180 / np.pi                                   # Convert from Radians to Degrees
      predicted_DOA = DOA_pred[:M][::-1]
      if(predicted_DOA.shape[0] < M):
        print("Cant estimate M sources - MUSIC")
      else:
        lossMUSIC = PRMSE(predicted_DOA, Y * 180 / np.pi)
        MUSIC_list.append(lossMUSIC)
  return np.mean(RootMUSIC_list), np.mean(MUSIC_list)


def PlotSpectrum(DeepRootMUSIC, DataSet_Rx_test, DataSet_x_test, Sys_Model):
  criterion = PRMSELoss()
  Data_Set_path = r"G:\My Drive\Thesis\\DeepRootMUSIC\Code\\DataSet"
  fig = plt.figure(figsize=(16, 12), dpi=80)
  PLOT_MUSIC = True
  PLOT_ROOT_MUSIC = True
  PLOT_DeepROOT_MUSIC = True

  DataSet_Rx_test = torch.utils.data.DataLoader(DataSet_Rx_test,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False)
  
  DataSet_x_test = torch.utils.data.DataLoader(DataSet_x_test,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False)
  
  model_based_platform = Model_Based_methods(Sys_Model)

  RootMUSIC_loss, MUSIC_loss = evaluate_model_based(DataSet_x_test, Sys_Model)
  DeepRootTest_loss = evaluate_model(DeepRootMUSIC, DataSet_Rx_test, criterion)      
  print("Deep Root-MUSIC Test loss = {}".format(DeepRootTest_loss))
  print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))
  print("MUSIC Test loss = {}".format(MUSIC_loss))
  print("\n--- Interpretability Stage ---\n")
  ############################
  ## model-based evaluation ##
  ############################

  for i,data in enumerate(DataSet_x_test):
      X, Y = data
      print("Real Angle:", Y * 180 / np.pi)
      X_modelbased = X[0]
      ## RootMUSIC predictions
      DOA_pred_RootMUSIC, roots, M, DOA_pred_all, roots_angels_all = model_based_platform.Classic_Root_MUSIC(X_modelbased, NUM_OF_SOURCES=True)
      lossRootMUSIC = PRMSE(DOA_pred_RootMUSIC, Y * 180 / np.pi)
      print("Root-MUSIC Estimated Angle:", DOA_pred_RootMUSIC)
      print("Root-MUSIC Loss:", lossRootMUSIC)
      
      ## MUSIC predictions
      DOA_pred_MUSIC, Spectrum, M = model_based_platform.Classic_MUSIC(X_modelbased, NUM_OF_SOURCES=M)
      DOA_pred = model_based_platform.angels[DOA_pred_MUSIC] * 180 / np.pi                                   # Convert from Radians to Deegres
      predicted_DOA = DOA_pred[:M][::-1]
      lossMUSIC = PRMSE(predicted_DOA, Y * 180 / np.pi)
      print("MUSIC Estimated Angle:", DOA_pred[:M])
      print("MUSIC Loss:", lossMUSIC)
      print("\n\n")
  
  ############################
  ##  Deep Root_MUSIC eval  ##
  ############################

  DeepRootMUSIC.eval()
  with torch.no_grad():
    for i,data in enumerate(DataSet_Rx_test):
        ## DeepRootMUSIC predictions
        Rx, DOA = data
        # Rx = Rx.to(device)
        # DOA = DOA.to(device)
        Y_pred, DOA_all, roots_deep = DeepRootMUSIC(Rx, DOA.shape[1])
        Deep_RootMUSIC_loss = criterion(Y_pred, DOA)
        # print(Y_pred * 180 / np.pi)
        DOA_all = DOA_all.detach().numpy()
        DOA_all = np.reshape(DOA_all, DOA_all.shape[1]) * 180 / np.pi
        # print("DOA_all", DOA_all)
        roots_deep = list(roots_deep.detach().numpy())
        # print("roots_deep", roots_deep)
        if (Deep_RootMUSIC_loss > 0):
          print("Real Angle:", DOA * 180 / np.pi)
          print("Deep Root-MUSIC Estimated Angle:", Y_pred * 180 / np.pi)
          print("Deep Root-MUSIC Loss:", Deep_RootMUSIC_loss)
          print("\n\n")

  if PLOT_MUSIC:
      ax1 = fig.add_subplot(131)
      ax1.set_title("Classic MUSIC")
      ax1.set_xlabel("Angels of Arrivels")
      ax1.set_ylabel("Spectrum Amplitude")
      ax1.plot(model_based_platform.angels * 180 / np.pi , Spectrum)
      ax1.plot(DOA_pred[:M], Spectrum[DOA_pred_MUSIC[:M]], "x")
  
  if PLOT_ROOT_MUSIC:
      ax2 = fig.add_subplot(132, polar=True)
      ax2.set_title("Classic Root MUSIC")
      ax2.set_xlabel("Angels of Arrivels")

      DOA_info = {}
      for i in range(len(DOA_pred_all)):
          DOA_info[DOA_pred_all[i]] = abs(roots[i])
      for angle, r in DOA_info.items():
          # print("Root-MUSIC: angle={}, r={}".format(angle,r))
          ax2.plot([0,angle * np.pi / 180],[0, r],marker='o')
  
  if PLOT_DeepROOT_MUSIC:
      ax3 = fig.add_subplot(133, polar=True)
      ax3.set_title("Deep Root MUSIC")
      ax3.set_xlabel("Angels of Arrivels")

      DOA_info = {}
      for i in range(len(DOA_all)):
          DOA_info[DOA_all[i]] = abs(roots_deep[i])

      for angle, r in DOA_info.items():
          # print("Deep Root-MUSIC :angle={}, r={}".format(angle,r))
          ax3.plot([0,angle * np.pi / 180],[0, r],marker='o')

  plt.show()