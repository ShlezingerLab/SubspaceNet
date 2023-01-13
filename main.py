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
plt.close('all')
os.system('cls||clear')

if __name__ == "__main__":
    device              = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    Main_path           = r"C:\Users\dorsh\OneDrive\Desktop\My Drive\Thesis\DeepRootMUSIC\Code"
    Main_Data_path      = Main_path + r"\\DataSet"
    saving_path         = Main_path + r"\\Weights"
    Simulations_path    = Main_path + r"\\Simulations"
    Data_Scenario_path  = r"\\LowSNR"

    Set_Overall_Seed()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    ############################
    ##        Commands        ##
    ############################
    SAVE_TO_FILE = False
    CREATE_DATA = True
    LOAD_DATA = False
    TRAIN_MODE = False
    SAVE_MODEL = False
    EVALUATE_MODE = True
    
    if(SAVE_TO_FILE):
        file_path = Simulations_path + r"\\Results\\Scores\\" + dt_string_for_save + r".txt"
        sys.stdout = open(file_path, "w")

    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    
    ############################
    ##    Data Parameters     ##
    ############################
    tau = 8
    N = 8
    M = 2
    T = 20
    SNR = 10
    nNumberOfSampels = 100
    Train_Test_Ratio = 1
    # scenario = "Broadband_OFDM"
    scenario = "NarrowBand"
    # scenario = "Broadband_simple"
    mode = "coherent"
    
    ############################
    ###   Create Data Sets   ###  
    ############################
    
    if CREATE_DATA:
        Set_Overall_Seed()
        Create_Training_Data = False
        Create_Testing_Data = True
        print("Creating Data...")
        if Create_Training_Data:
        ## Training Datasets
            DataSet_x_train, DataSet_Rx_train, _ = CreateDataSetCombined(
                                    scenario= scenario,
                                    mode= mode,
                                    N= N, M= M , T= T,
                                    Sampels_size = nNumberOfSampels,
                                    tau = tau,
                                    Save = True,
                                    DataSet_path = Main_Data_path + Data_Scenario_path + r"\\TrainingData",
                                    True_DOA = None,
                                    SNR = SNR)
        if Create_Testing_Data:
        ## Test Datasets
            DataSet_x_test, DataSet_Rx_test, Sys_Model = CreateDataSetCombined(
                                    scenario = scenario,
                                    mode = mode,
                                    N= N, M= M , T= T,
                                    Sampels_size = int(Train_Test_Ratio * nNumberOfSampels),
                                    tau = tau,
                                    Save = True,
                                    DataSet_path= Main_Data_path + Data_Scenario_path + r"\\TestData",
                                    True_DOA = None,
                                    SNR = SNR)

    ############################
    ###    Load Data Sets    ###
    ############################
    
    if LOAD_DATA:
        train_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, nNumberOfSampels, M, N, T, SNR)
        test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(Train_Test_Ratio * nNumberOfSampels), M, N, T, SNR)
        # train_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, nNumberOfSampels, M, N, T)
        # test_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(Train_Test_Ratio * nNumberOfSampels), M, N, T)

        DataSet_Rx_train = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + train_details_line)
        DataSet_Rx_test  = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + test_details_line)
        DataSet_x_test   = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + test_details_line)
        Sys_Model        = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + test_details_line)


    ############################
    ###    Training stage    ###
    ############################
    
    if TRAIN_MODE:
        # Training parameters
        optimal_gamma_val = 1
        optimal_bs = 2048
        optimal_lr = 0.001
        optimal_step = 1
        epochs = 80

        Test_losses = []
        train_loss_lists = []
        validation_loss_lists = []

        train_curves = []
        validation_curves = []

        fig = plt.figure(figsize=(8, 6), dpi=80)
        
        ############################
        ###    Run Simulation    ###
        ############################
 
        print("\n--- New Simulation ---\n")
        # print("Description: Simulation of broadband sources within range [0-500] Hz with T = {}, Tau = {}, SNR = {}, {} sources".format(T, tau, SNR, mode))
        print("Description: Simulation of broadband sources within range [0-500] Hz with T = {}, Tau = {}, SNR = {}, {} sources".format(T, tau, SNR, mode))
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
        
        model, loss_train_list, loss_valid_list, Test_loss = Run_Simulation(
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
                        load_flag = False,
                        loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T),
                        Plot = False,
                        DataSetModelBased = DataSet_x_test)
        
        # Save model weights
        if SAVE_MODEL:
            torch.save(model.state_dict(), saving_path + r"/Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T))
        
        train_loss_lists.append(loss_train_list)
        validation_loss_lists.append(loss_valid_list)
        Test_losses.append(Test_loss)
        
        # Plotting train & validation curves
        plt.plot(range(epochs), loss_train_list, label="tr {}".format(optimal_lr))
        plt.plot(range(epochs), loss_valid_list, label="vl {}".format(optimal_lr))
        plt.title("Learning Curves: Loss per Epoch - different Learning Rates")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=0)
        
        # Plots saving
        if(SAVE_TO_FILE):
            plt.savefig(Simulations_path + r"\\Results\\Plots\\" + dt_string_for_save + r".png")
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
        
        test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(Train_Test_Ratio * nNumberOfSampels), M, N, T, SNR)
        # test_details_line = '_{}_{}_{}_M={}_N={}_T={}.h5'.format(scenario, mode, int(Train_Test_Ratio * nNumberOfSampels), M, N, T)

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
        loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
        
        if scenario.startswith("Broadband"):
            model = Deep_Root_Net_Broadband(tau=tau, ActivationVal=0.5)  
            loading_path = saving_path + r"\Final_models" + r"\BroadBand" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(2, mode, tau, SNR, 200)
        elif T == 200 and SNR == 10 and M==2: 
            model = Deep_Root_Net(tau=tau, ActivationVal=0.5)                                         
        else:
            model = Deep_Root_Net_AntiRectifier(tau=tau, ActivationVal=0.5) 

        
        # Load it to the specified device, either gpu or cpu
        model = model.to(device)
        if torch.cuda.is_available() == False:
                        model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
                        
        criterion = PRMSELoss() #define loss criterion
        
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
  
        DeepRootTest_loss = evaluate_model(model, DataSet_Rx_test, criterion)      
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
        plt.show()
        print("end")