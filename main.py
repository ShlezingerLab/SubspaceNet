import sys
import bs4
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
from useful_func import * 

warnings.simplefilter("ignore")
plt.close('all')
os.system('cls||clear')

if __name__ == "__main__":
    Set_Overall_Seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    Main_path = r"G:\\My Drive\\Thesis\\DeepRootMUSIC\\Code"
    Main_Data_path = r"G:\\My Drive\\Thesis\\DeepRootMUSIC\\Code\\DataSet"
    Data_Scenario_path = r"\\LowSNR"
    saving_path = r"G:\My Drive\Thesis\\DeepRootMUSIC\Code\\Weights\Models"
    Simulations_path = r"G:\My Drive\Thesis\\DeepRootMUSIC\\Code\\Simulations"

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    
    ############################
    ##        Commands        ##
    ############################
    Save_to_File = True
    CreateData = False
    
    if(Save_to_File):
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
    T = 200
    nNumberOfSampels = 100000
    Train_Test_Ratio = 0.05
    scenario = "NarrowBand"
    mode = "non-coherent"
    SNR = -6
    
    ############################
    ###   Create Data Sets   ###
    ############################
    
    if CreateData:
        Set_Overall_Seed()
        Create_Training_Data = True
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
    
    train_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, nNumberOfSampels, M, N, T, SNR)
    test_details_line = '_{}_{}_{}_M={}_N={}_T={}_SNR={}.h5'.format(scenario, mode, int(Train_Test_Ratio * nNumberOfSampels), M, N, T, SNR)

    DataSet_Rx_train = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + train_details_line)
    DataSet_Rx_test = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx" + test_details_line)
    DataSet_x_test = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x" + test_details_line)
    Sys_Model = Read_Data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model" + test_details_line)

    ############################
    ##   Training parameters  ##
    ############################

    optimal_gamma_val = 0.01
    optimal_bs = 2048
    lr_list = [0.00001, 0.0001, 0.000001] # maybe optimal to examine for 0.01
    optimal_step = 80
    epochs = 15
    
    ############################
    ###    Run Simulations   ###
    ############################
    
    Test_losses = []
    train_loss_lists = []
    validation_loss_lists = []

    train_curves = []
    validation_curves = []

    fig = plt.figure(figsize=(8, 6), dpi=80)
    print("Description: Loaded lr simulation of 2 Low SNR = {} non-coherent sources".format(SNR))
    # print("Description: Loading Best results simulation of {} LOW SNR with Lr = {} and Batch Size {}".format(M, optimal_lr, 1500))
    
    for lr in lr_list:
        print("\n--- New Simulation ---\n")
        print("Simulation parameters:")
        print("Learning Rate = {}".format(lr))
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
                        lr_val = lr,
                        Schedular = True,
                        weight_decay_val = 1e-9,
                        step_size_val = optimal_step,
                        gamma_val = optimal_gamma_val,
                        num_epochs = epochs,
                        model_name= "model_tau=8_M=2_70Ksampels_LowSNR_{}".format(SNR),
                        Bsize = optimal_bs,
                        Sys_Model = Sys_Model,
                        load_flag = True,
                        loading_path = saving_path + r"/model_tau=8_M=2_70Ksampels_LowSNR_-613_07_2022_21_49",
                        Plot = False,
                        DataSetModelBased = DataSet_x_test)
        
        train_loss_lists.append(loss_train_list)
        validation_loss_lists.append(loss_valid_list)
        Test_losses.append(Test_loss)
        
        plt.plot(range(epochs), loss_train_list, label="tr {}".format(lr))
        plt.plot(range(epochs), loss_valid_list, label="vl {}".format(lr))

    plt.title("Learning Curves: Loss per Epoch - different Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=0)
    
    if(Save_to_File):
        plt.savefig(Simulations_path + r"\\Results\\Plots\\" + dt_string_for_save + r".png")
    else:
        plt.show()
        os.system("pause")