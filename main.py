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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    Main_path = r"G:\\My Drive\\Thesis\\DeepRootMUSIC\\Code"
    Main_Data_path = r"G:\\My Drive\\Thesis\\DeepRootMUSIC\\Code\\DataSet"
    Data_Scenario_path = r"\\LowSNR"
    # saving_path = r"G:\My Drive\Thesis\DeepRootMUSIC\Code\Weights\Models"
    saving_path = r"G:\My Drive\Thesis\DeepRootMUSIC\Code\Weights"
    Simulations_path = r"G:\My Drive\Thesis\\DeepRootMUSIC\\Code\\Simulations"

    Set_Overall_Seed()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    ############################
    ##        Commands        ##
    ############################
    SAVE_TO_FILE = False
    CREATE_DATA = False
    LOAD_DATA = True
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
    nNumberOfSampels = 100000
    Train_Test_Ratio = 0.05
    scenario = "NarrowBand"
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
                                    Save = False,
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
        optimal_lr = 0.00001
        optimal_step = 80
        epochs = 1

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
        print("Description: Simulation of closely spaced sources with T = {}, Tau = {}, SNR = {}, {} sources".format(T, tau, SNR, mode))
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
            torch.save(model.state_dict(), saving_path + r"/Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}_MSE".format(M, mode, tau, SNR, T))
        
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
        # loading_path = saving_path + r"\Final_models" + r"\model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
        loading_path = saving_path + r"\Final_models" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
        model = Deep_Root_Net_AntiRectifier(tau=tau, ActivationVal=0.5)  
        # model = Deep_Root_Net(tau=tau, ActivationVal=0.5)                                         
        
        # Load it to the specified device, either gpu or cpu
        model = model.to(device)         
        if torch.cuda.is_available() == False:
                        model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
        
        Losses = PlotSpectrum(model, DataSet_Rx_test, DataSet_x_test, Sys_Model)
        RootMUSIC_loss.append(Losses[0])
        MUSIC_loss.append(Losses[1])
        SPS_RootMUSIC_loss.append(Losses[2])
        SPS_MUSIC_loss.append(Losses[3])
        DeepRootTest_loss.append(Losses[4])
            
        print("MUSIC_{} = np.array(".format(mode), MUSIC_loss, ")")
        print("RootMUSIC_{} = np.array(".format(mode) , RootMUSIC_loss, ")")
        print("SPS_RootMUSIC_{} = np.array(".format(mode), SPS_RootMUSIC_loss, ")")
        print("SPS_MUSIC_{} = np.array(".format(mode), SPS_MUSIC_loss, ")")
        print("DeepRootMUSIC_{} = np.array(".format(mode), DeepRootTest_loss, ")")