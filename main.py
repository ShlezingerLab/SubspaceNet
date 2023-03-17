"""Subspace-Net 
Details
----------
Name: main.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This script allows the user to run simulation of the proposed algorithm,
by wrapping all the required procedures, calling the following functions:
    * create_dataset: For creating training and testing datasets 
    * run_simulation: For training DR-MUSIC model
    * evaluate_model: For evaluating subspace hybrid models

This script requires that requirements.txt will be installed within the Python
environment you are running this script in.

"""
###############
#   Imports   #
###############

import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from System_Model import *
from Signal_creation import *
from DataLoaderCreation import *
from EvaluationMesures import *
from methods import *
from models import *
from Run_Simulation import *
from utils import * 

warnings.simplefilter("ignore")
os.system('cls||clear')
plt.close('all')

if __name__ == "__main__":
    # Set relevant paths
    Main_path           = r"C:\Users\dorsh\Deep RootMUSIC\Code"
    Main_Data_path      = Main_path + "\DataSet"
    saving_path         = Main_path + "\Weights"
    simulations_path    = Main_path + "\Simulations"
    Data_Scenario_path  = r"\Calibration_array"
    
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    # Define compilation device and set seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    set_unified_seed()
    
    ############################
    ##   Operation commands   ##
    ############################
    SAVE_TO_FILE = False            # Flag for Whether saving results to file or present them over CMD
    CREATE_DATA = True             # Flag for creating new data
    LOAD_DATA = True                # Flag for loading data from dataset
    TRAIN_MODE = True               # Flag for applying training operation 
    SAVE_MODEL = False              # Flag for saving tuned model
    EVALUATE_MODE = True            # Flag for evaluating desired algorithms
    
    if SAVE_TO_FILE:
        file_path = simulations_path + r"\\Results\\Scores\\" + dt_string_for_save + r".txt"
        sys.stdout = open(file_path, "w")

    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    
    ############################
    #    Parameters setting   #
    ############################
    
    # System model parameters
    tau = 8                         # Number of lags
    N = 8                           # Number of sensors
    M = 2                           # number of sources
    T = 200                         # Number of observations, ideal = 200 or above
    SNR = 10                        # Signal to noise ratio, ideal = 10 or above
    eta = 0.0625                    # Deviation from sensor location, normalized by wavelength, ideal = 0
    scenario = "NarrowBand"         # signals type, options: "NarrowBand", "Broadband_OFDM", "Broadband_simple"
    mode = "non-coherent"           # signals nature, options: "non-coherent", "coherent"
    
    # simulation parameters
    samples_size = 10            # Overall dateset size 
    train_test_ratio = 0.05         # training and testing datasets ratio 
    
    ############################
    #     Create Data Sets     #  
    ############################
    
    if CREATE_DATA:
        set_unified_seed()
        Create_Training_Data = True # Flag for creating training data
        Create_Testing_Data = True  # Flag for creating test data
        
        print("Creating Data...")
        if Create_Training_Data:
            ## Training Datasets
            DataSet_x_train, DataSet_Rx_train, _ = create_dataset(
                                    scenario= scenario,
                                    mode= mode,
                                    N= N, M= M , T= T,
                                    Sampels_size = samples_size,
                                    tau = tau,
                                    Save = True,
                                    DataSet_path = Main_Data_path + Data_Scenario_path + r"\TrainingData",
                                    True_DOA = None,
                                    SNR = SNR,
                                    eta=eta)
        if Create_Testing_Data:
            ## Test Datasets
            DataSet_x_test, DataSet_Rx_test, Sys_Model = create_dataset(
                                    scenario = scenario,
                                    mode = mode,
                                    N= N, M= M , T= T,
                                    Sampels_size = int(train_test_ratio * samples_size),
                                    tau = tau,
                                    Save = True,
                                    DataSet_path= Main_Data_path + Data_Scenario_path + r"\TestData",
                                    True_DOA = None,
                                    SNR = SNR,
                                    eta = eta)

    ############################
    ###    Load Data Sets    ###
    ############################
    
    if LOAD_DATA:
        train_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}.h5'.format(scenario, mode, samples_size, M, N, T, SNR, str(eta).replace(",", ""))
        test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR, str(eta).replace(",", ""))
        # train_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, samples_size, M, N, T)
        # test_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T)

        DataSet_Rx_train = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + train_details_line)
        DataSet_x_train  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_x"   + train_details_line)
        DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + test_details_line)
        DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + test_details_line)
        Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + test_details_line)


    ############################
    ###    Training stage    ###
    ############################
    
    if TRAIN_MODE:
        # Training aided parameters
        optimal_lr = 0.00001        # Learning rate value
        optimal_bs = 64             # Batch size value
        epochs = 30                 # Number of epochs
        optimal_step = 1            # Number of steps for learning rate decay iteration
        optimal_gamma_val = 1       # learning rate decay value

        # list containers declaration
        Test_losses = []
        train_loss_lists = []
        validation_loss_lists = []
        train_curves = []
        validation_curves = []

        ############################
        ###    Run Simulation    ###
        ############################
 
        print("\n--- New Simulation ---\n")
        # print("Description: Simulation of broadband sources within range [0-500] Hz with T = {}, Tau = {}, SNR = {}, {} sources".format(T, tau, SNR, mode))
        print("Description: Simulation with constant {} deviation in sensors location, T = {}, SNR = {}, {} sources".format(eta, T, SNR, mode))
        print("Simulation parameters:")
        print("Learning Rate = {}".format(optimal_lr))
        print("Batch Size = {}".format(optimal_bs))
        print("SNR = {}".format(SNR))
        print("scenario = {}".format(scenario))
        print("mode = {}".format(mode))
        print("Gamma Value = {}".format(optimal_gamma_val))
        print("Step Value = {}".format(optimal_step))
        print("Epochs = {}".format(epochs))
        print("Tau = {}".format(tau))
        print("Observations = {}".format(T))
        print("spacing deviation (eta) = {}".format(eta))
        
        model, loss_train_list, loss_valid_list, Test_loss = run_simulation(
                        Model_Train_DataSet = DataSet_Rx_train,
                        Model_Test_DataSet = DataSet_Rx_test,
                        tau = tau,
                        N = N,
                        optimizer_name = "Adam",
                        lr_val = optimal_lr,
                        Schedular = True,
                        weight_decay_val = 1e-9,
                        step_size_val = optimal_step,
                        gamma_val = optimal_gamma_val,
                        num_epochs = epochs,
                        model_name= "model_tau=2_M=2_100Samples_SNR_{}_T=2_just_a_test".format(SNR),
                        Bsize = optimal_bs,
                        Sys_Model = Sys_Model,
                        load_flag = True,
                        # loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T),
                        # loading_path = saving_path + r"\Final_models" + r"/model_16_03_2023_23_54",
                        loading_path = saving_path + r"\Models" + r"/model_16_03_2023_23_54",
                        Plot = False,
                        DataSetModelBased = DataSet_x_test)
        
        # Save model weights
        if SAVE_MODEL:
            torch.save(model.state_dict(), saving_path + r"/Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}_eta={}".format(M, mode, tau, SNR, T, eta))
        
        train_loss_lists.append(loss_train_list)
        validation_loss_lists.append(loss_valid_list)
        Test_losses.append(Test_loss)
        
        # Plotting train & validation curves
        fig = plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(range(epochs), loss_train_list, label="tr {}".format(optimal_lr))
        plt.plot(range(epochs), loss_valid_list, label="vl {}".format(optimal_lr))
        plt.title("Learning Curves: Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=0)
        
        # Plots saving
        if SAVE_TO_FILE:
            plt.savefig(simulations_path + r"\\Results\\Plots\\" + dt_string_for_save + r".png")
        else:
            plt.show()
    

    ############################
    ###   Evaluation stage   ###
    ############################
    
    if EVALUATE_MODE:
        RootMUSIC_loss = []
        MUSIC_loss = []
        SPS_RootMUSIC_loss = []
        SPS_MUSIC_loss = []
        DeepRootTest_loss = []
        if not CREATE_DATA:
            test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR, eta)
            # test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR)
            # test_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T)
            DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx" + test_details_line)
            DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"  + test_details_line)
            Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"  + test_details_line)
        
        print("SNR = {}".format(SNR))
        print("scenario = {}".format(scenario))
        print("mode = {}".format(mode))
        print("Observations = {}".format(T))
        

        ############################
        ###    Load Data Set     ###
        ############################
        
        if not TRAIN_MODE:
            if scenario.startswith("Broadband"):
                loading_path = saving_path + r"\Final_models" + r"\BroadBand" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(2, mode, tau, SNR, 200)
            else:
                # loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
                # loading_path = saving_path + r"\Final_models" + r"/model_04_03_2023_00_32"
                loading_path = saving_path + r"\Models" + r"/model_26_02_2023_23_41"

            if T == 200 and SNR == 10 and (M==2 or M == 3) and eta == 0: 
                model = Deep_Root_Net(tau=tau, ActivationVal=0.5)                                         
            else:
                model = Deep_Root_Net_AntiRectifier(tau=tau) 
                    
            # Load the model to the specified device, either gpu or cpu
            model = model.to(device)
            try:
                if torch.cuda.is_available() == False:
                    model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
            except:
                print("No loaded weights found")
                pass
            
        criterion = PRMSELoss() # define loss criterion
        Data_Set_path = Main_path + r"\\DataSet"
        DataSet_Rx_test = torch.utils.data.DataLoader(DataSet_Rx_test,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)
        
        DataSet_x_test = torch.utils.data.DataLoader(DataSet_x_test,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)
        
        mb = ModelBasedMethods(Sys_Model)
  
        DeepRootTest_loss = evaluate_model(model, DataSet_Rx_test, criterion, plot_spec= True)     
        print("Deep Root-MUSIC Test loss = {}".format(DeepRootTest_loss))                
        loss_hybrid_MUSIC, loss_hybrid_ESPRIT = evaluate_hybrid_model(model, DataSet_Rx_test, Sys_Model)
        print("hybrid MUSIC Test loss = {}".format(loss_hybrid_MUSIC))
        print("hybrid ESPRIT Test loss = {}".format(loss_hybrid_ESPRIT))

        losses = evaluate_model_based(DataSet_x_test, Sys_Model)
        
        if scenario.startswith("Broadband"):
            BB_MUSIC_loss, MUSIC_loss, ESPRIT_loss = losses
            print("BB MUSIC Test loss = {}".format(BB_MUSIC_loss))
            print("MUSIC Test loss = {}".format(MUSIC_loss))
            print("ESPRIT Test loss = {}".format(ESPRIT_loss))
        
        elif scenario.startswith("NarrowBand"):
            RootMUSIC_loss, MUSIC_loss, SPS_RootMUSIC_loss, SPS_MUSIC_loss, ESPRIT_loss= losses
            print("MUSIC Test loss = {}".format(MUSIC_loss))
            print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))
            print("ESPRIT Test loss = {}".format(ESPRIT_loss))
            print("Spatial Smoothing Root-MUSIC Test loss = {}".format(SPS_RootMUSIC_loss))
            print("Spatial Smoothing MUSIC Test loss = {}".format(SPS_MUSIC_loss))
        print("end")
    plt.legend()
    plt.show()