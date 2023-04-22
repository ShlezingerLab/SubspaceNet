"""Subspace-Net main script 
    Details
    ------------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 17/03/23

    Purpose
    ------------
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
from criterions import *
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
    # Data_Scenario_path  = r"\CoherentSourcesWithGap"
    Data_Scenario_path  = r"\LowSNR"
    
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
    commands = {"SAVE_TO_FILE"  : True,    # Saving results to file or present them over CMD
                "CREATE_DATA"   : False,    # Creating new data
                "LOAD_DATA"     : False,     # Loading data from dataset 
                "TRAIN_MODE"    : False,    # Applying training operation
                "SAVE_MODEL"    : False,    # Saving tuned model
                "EVALUATE_MODE" : True}     # Evaluating desired algorithms
                
    if commands["SAVE_TO_FILE"]:
        file_path = simulations_path + r"\\Results\\Scores\\" + dt_string_for_save + r".txt"
        sys.stdout = open(file_path, "w")

    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    
    ############################
    #    Parameters setting   #
    ############################
    # for SNR in [5, 6, 7, 8, 9, 10]:
    # System model parameters
    tau = 8                     # Number of lags
    N = 8                       # Number of sensors
    M = 2                       # number of sources
    T = 500                     # Number of observations, ideal = 200 or above
    SNR = 10                    # Signal to noise ratio, ideal = 10 or above
    
    ## Signal parameters
    scenario = "Broadband_OFDM"     # signals type, options: "NarrowBand", "Broadband_OFDM", "Broadband_simple"
    mode = "coherent"           # signals nature, options: "non-coherent", "coherent"
    
    ## Array mis-calibration values
    eta = 0                     # Deviation from sensor location, normalized by wavelength, ideal = 0
    geo_noise_var = 0           # Added noise for sensors response
    
    # simulation parameters
    samples_size = 60000         # Overall dateset size 
    train_test_ratio = 0.05     # training and testing datasets ratio 
    
    ############################
    #     Create Data Sets     #  
    ############################
    
    if commands["CREATE_DATA"]:
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
                                    eta=eta,
                                    geo_noise_var = geo_noise_var)
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
                                    eta = eta,
                                    geo_noise_var = geo_noise_var)

    ############################
    ###    Load Data Sets    ###
    ############################
    
    if commands["LOAD_DATA"]:
        try:
            TRAIN_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, samples_size, M, N,
                                    T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))
            TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                    M, N, T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))

            DataSet_Rx_train = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + TRAIN_DATA_PATH)
            DataSet_x_train  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_x"   + TRAIN_DATA_PATH)
            DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + TEST_DATA_PATH)
            DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + TEST_DATA_PATH)
            Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)
        except:
            TRAIN_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, samples_size, M, N, T, SNR)
            TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR)
            DataSet_Rx_train = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + TRAIN_DATA_PATH)
            DataSet_x_train  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_x"   + TRAIN_DATA_PATH)
            DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + TEST_DATA_PATH)
            DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + TEST_DATA_PATH)
            Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)


    ############################
    ###    Training stage    ###
    ############################
    
    if commands["TRAIN_MODE"]:
        # Training aided parameters
        optimal_lr = 0.001          # Learning rate value
        optimal_bs = 2048           # Batch size value
        epochs = 80                 # Number of epochs
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
        # print("Description: Simulation with constant {} deviation in sensors location, T = {}, SNR = {}, {} sources".format(eta, T, SNR, mode))
        print("Description: Simulation geometry mis-matches with added noise to array response", end=" ")
        print(f"variance = {geo_noise_var} , T = {T}, SNR = {SNR}, {mode} sources")
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
        print("Spacing deviation (eta) = {}".format(eta))
        print("Geometry noise variance = {}".format(geo_noise_var))
        
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
                        loading_path = saving_path + r"\Models" + r"/model_11_04_2023_03_48",
                        # loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(2, mode, tau, 10, 20),
                        Plot = False,
                        DataSetModelBased = DataSet_x_test)
        
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(model.state_dict(), saving_path + r"/Final_models" \
                       + r"/model_M={}_{}_Tau={}_SNR={}_T={}_eta={}".format(M, mode, tau, SNR, T, eta, geo_noise_var))
        
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
        if commands["SAVE_TO_FILE"]:
            plt.savefig(simulations_path + r"\\Results\\Plots\\" + dt_string_for_save + r".png")
        else:
            plt.show()
    

    ############################
    ###   Evaluation stage   ###
    ############################
    
    if commands["EVALUATE_MODE"]:
        augmented_methods = ["music", "esprit"]
        subspace_methods = ["music", "esprit", "root-music"]
        # RootMUSIC_loss = []
        MUSIC_loss = []
        # SPS_RootMUSIC_loss = []
        SPS_MUSIC_loss = []
        DeepRootTest_loss = []
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            # TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), 3, N, T)
            # TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR)
            # TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR, eta)
            TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                                                                               M, N, T, SNR, str(eta).replace(",", ""),
                                                                                               str(geo_noise_var).replace(",", ""))
            # TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T, SNR)
            # TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(train_test_ratio * samples_size), M, N, T)
            DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx" + TEST_DATA_PATH)
            DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"  + TEST_DATA_PATH)
            Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"  + TEST_DATA_PATH)
        
        print("SNR = {}".format(SNR))
        print("scenario = {}".format(scenario))
        print("mode = {}".format(mode))
        print("Observations = {}".format(T))
        

        ############################
        ###    Load Data Set     ###
        ############################
        
        if not commands["TRAIN_MODE"]:
            if scenario.startswith("Broadband"):
                # loading_path = saving_path + r"\Final_models" + r"\BroadBand" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
                loading_path = saving_path + r"\Models" + r"\model_11_04_2023_03_48"
                # loading_path = r"C:\Users\dorsh\Deep RootMUSIC\Code\Weights\Models\model_11_04_2023_03_48"
            else:
                pass
                # loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
                # loading_path = saving_path + r"\Models" + r"/model_27_03_2023_21_46"

            if T in [100, 200] and SNR == 10 and (M in [2, 3, 4]) and eta == 0 and geo_noise_var == 0 and not scenario.startswith("Broadband"): 
                # model = Deep_Root_Net(tau=tau, ActivationVal=0.5)                                         
                model = Deep_Root_Net_AntiRectifier(tau=tau)                                    
            else:
                model = Deep_Root_Net_AntiRectifier(tau=tau) 
                    
            # Load the model to the specified device, either gpu or cpu
            model = model.to(device)
            try:
                if torch.cuda.is_available() is False:
                    model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
            except:
                print("No loaded weights found")
                pass
            
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

        criterion = RMSPELoss() # define loss criterion
        hybrid_criterion = RMSPE
        # criterion = MSPELoss() # define loss criterion
        # hybrid_criterion = MSPE
        # Evaluate SubspaceNet augmented methods
        DeepRootTest_loss = evaluate_model(model, DataSet_Rx_test, criterion=criterion, plot_spec= True)
        print(f"Deep Root-MUSIC Test loss = {DeepRootTest_loss}")
        for algorithm in augmented_methods:
            hybrid_loss = evaluate_hybrid_model(model, DataSet_Rx_test, Sys_Model, criterion = hybrid_criterion, algorithm = algorithm)
            print("hybrid {} test loss = {}".format(algorithm, hybrid_loss))

        losses = evaluate_model_based(DataSet_x_test, Sys_Model, criterion=hybrid_criterion)
        
        if scenario.startswith("Broadband"):
            BB_MUSIC_loss, MUSIC_loss, ESPRIT_loss, RootMUSIC_loss = losses
            print("BB MUSIC Test loss = {}".format(BB_MUSIC_loss))
            print("MUSIC Test loss = {}".format(MUSIC_loss))
            print("ESPRIT Test loss = {}".format(ESPRIT_loss))
            print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))

        elif scenario.startswith("NarrowBand"):
            (RootMUSIC_loss, MUSIC_loss, SPS_RootMUSIC_loss, SPS_MUSIC_loss, ESPRIT_loss, SPS_ESPRIT_loss) = losses
            print("MUSIC Test loss = {}".format(MUSIC_loss))
            print("Root-MUSIC Test loss = {}".format(RootMUSIC_loss))
            print("ESPRIT Test loss = {}".format(ESPRIT_loss))
            print("Spatial Smoothing Root-MUSIC Test loss = {}".format(SPS_RootMUSIC_loss))
            print("Spatial Smoothing MUSIC Test loss = {}".format(SPS_MUSIC_loss))
            print("Spatial Smoothing ESPRIT Test loss = {}".format(SPS_ESPRIT_loss))
        print("end")
    # plt.legend()
    # plt.show()