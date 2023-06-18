"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 17/03/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * run_simulation: For training DR-MUSIC model
        * evaluate_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.signal_creation import *
from src.data_handler import *
from src.criterions import *
from src.methods import *
from src.models import *
from src.simulation_handler import *
from src.utils import simulation_summary
# from src.utils import * 
from pathlib import Path

# Initialization
warnings.simplefilter("ignore")
os.system('cls||clear')
plt.close('all')

if __name__ == "__main__":
    # Set relevant paths
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "LowSNR"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    
    # Operations commands
    commands = {"SAVE_TO_FILE"  : False,    # Saving results to file or present them over CMD
                "CREATE_DATA"   : True,     # Creating new data
                "LOAD_DATA"     : False,    # Loading data from dataset 
                "TRAIN_MODE"    : True,     # Applying training operation
                "SAVE_MODEL"    : False,    # Saving tuned model
                "EVALUATE_MODE" : True}     # Evaluating desired algorithms
    
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = simulations_path / "results" / Path("scores" + dt_string_for_save + ".txt")
        sys.stdout = open(file_path, "w")
    ## Define DNN model
    model_type = "SubspaceNet" # Set model type, options: "DA-MUSIC", "DeepCNN"  
    tau = 8 # Number of lags, relevant for "SubspaceNet" model
    ## Define system model parameters
    N = 8       # Number of sensors
    M = 2       # number of sources
    T = 200     # Number of observations, ideal >= 200
    SNR = 10    # Signal to noise ratio, ideal = 10
    ## Define signal parameters
    scenario = "NarrowBand" # signals type, options: "NarrowBand", "Broadband_OFDM", "Broadband_simple"
    mode = "non-coherent"   # signals nature, options: "non-coherent", "coherent"
    ## Array calibration values
    eta = 0                 # Deviation from sensor location, normalized by wavelength, ideal = 0
    geo_noise_var = 0       # Added noise for sensors response
    ## Simulation parameters
    samples_size = 10       # Overall dateset size
    train_test_ratio = 0.1  # training and testing datasets ratio 
    simulation_filename = f"{model_type}_M={M}_T={T}_SNR_{SNR}_tau={tau}_{scenario}_{mode}_eta={eta}_sv_noise={geo_noise_var}"
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Initialize seed 
    set_unified_seed()
    # Datasets creation
    if commands["CREATE_DATA"]:
        set_unified_seed()
        ## Define which datasets to generate
        create_training_data = True # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset 
            train_dataset, _, _ = create_dataset(scenario= scenario, mode= mode, N= N, M= M , T= T,\
                            samples_size = samples_size, tau = tau, model_type = model_type,\
                            Save = False, datasets_path = datasets_path / "TrainingData",\
                            true_doa = None, SNR = SNR, eta=eta, geo_noise_var = geo_noise_var, phase = "train")
        if create_testing_data:
            # Generate test dataset
            test_dataset, generic_test_dataset, samples_model = create_dataset(scenario = scenario, mode = mode,\
                            N= N, M= M , T= T, samples_size = int(train_test_ratio * samples_size), tau = tau,\
                            model_type = model_type, Save = False, datasets_path = datasets_path / "TestData",
                            true_doa =  None, SNR = SNR, eta = eta, geo_noise_var = geo_noise_var, phase = "test")
    # Datasets loading
    if commands["LOAD_DATA"]:
        train_dataset, test_dataset, generic_test_dataset, samples_model = load_datasets(model_type,\
                        scenario, mode, samples_size, M, N, T, SNR, eta, geo_noise_var,\
                        datasets_path=datasets_path, is_training = True)

    ### Training stage ###
    if commands["TRAIN_MODE"]:
        # Training aided parameters
        optimal_lr = 0.00001         # Learning rate value
        optimal_bs = 256            # Batch size value
        epochs = 50                # Number of epochs
        optimal_step = 100          # Number of steps for learning rate decay iteration
        optimal_gamma_val = 0.9      # learning rate decay value
        weight_decay_val = 1e-9
        # list containers declaration
        Test_losses = []
        train_loss_lists = []
        validation_loss_lists = []
        train_curves = []
        validation_curves = []
        # Print simulation summary details
        simulation_summary(model_type=model_type, M=M, N=N, T=T, SNR=SNR, scenario=scenario,\
                            mode=mode, eta=eta, geo_noise_var=geo_noise_var, optimal_lr=optimal_lr,\
                            weight_decay_val=weight_decay_val, batch_size=optimal_bs,\
                            optimal_gamma_val=optimal_gamma_val, optimal_step=optimal_step, epochs=epochs,\
                            phase="training", tau=tau)
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list, Test_loss = run_simulation(
                        train_dataset = train_dataset, test_dataset = test_dataset,
                        tau = tau, optimizer_name = "Adam", lr_val = optimal_lr, schedular = True,
                        weight_decay_val = weight_decay_val, step_size_val = optimal_step,
                        gamma_val = optimal_gamma_val, num_epochs = epochs, Plot = False,
                        model_name= simulation_filename, model_type = model_type,
                        batch_size = optimal_bs, system_model = samples_model,
                        load_flag = False, loading_path = saving_path / Path(simulation_filename))
        # Generate training and validation loss curves
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
        
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(model.state_dict(), saving_path / Path(simulation_filename))
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(simulations_path / "results" / "plots" / Path(dt_string_for_save + r".png"))
        else:
            plt.show()
    
    ### Evaluation stage ###
    if commands["EVALUATE_MODE"]:
        ## Generate benchmarks and augmentations
        if model_type.startswith("SubspaceNet"):
            # Set the model-based methods to augment via SubspaceNet
            augmented_methods = ["music", "mvdr", "esprit"]
        else:
            augmented_methods = []
        # Model-base methods for comparison
        subspace_methods = ["esprit", "music", "r-music",\
                        "sps-r-music", "sps-esprit", "sps-music", "bb-music"]
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(model_type,\
                            scenario, mode, samples_size, M, N, T, SNR, eta, geo_noise_var,\
                            datasets_path=datasets_path)
        ## simulation summary details
        simulation_summary(model_type=model_type, M=M, N=N, T=T, SNR=SNR, scenario=scenario,\
                            mode=mode, eta=eta, geo_noise_var=geo_noise_var, phase="evaluation")
        
        # Define loss measure
        loss_measure = "rmse"
        print(f"Loss measure = {loss_measure}")

        
        if not commands["TRAIN_MODE"]:
            # Load trained model
            loading_path =  saving_path / Path(simulation_filename)

            # Assign the desired DNN-model for evaluation
            if model_type.startswith("DeepCNN"):
                model = DeepCNN(N, 361)
            elif model_type.startswith("SubspaceNet"):
                model = SubspaceNet(tau=tau, M=M)                                         
            elif model_type.startswith("DA-MUSIC"):
                model = DeepAugmentedMUSIC(N=N, T=T, M=M)

            # Load the model to the specified device, either gpu or cpu
            model = model.to(device)
            try:
                if torch.cuda.is_available() is False:
                    model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
            except:
                raise Exception(("cannot load desired model"))
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1,\
                        shuffle=False, drop_last=False)
        generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset, batch_size=1,\
                        shuffle=False, drop_last=False)

        # TODO(DOR 16.06.23): continue from here
        mb_methods = ModelBasedMethods(samples_model)
        # Define loss measure for evaluation
        if loss_measure.startswith("rmse"):
            criterion = RMSPELoss() # define loss criterion
            hybrid_criterion = RMSPE
        elif loss_measure.startswith("mse"):
            criterion = MSPELoss() # define loss criterion
            hybrid_criterion = MSPE
            
        # Evaluate SubspaceNet augmented methods
        PLOT_SPECTRUM = False
        figures = {"music"  : {"fig" : None, "ax" : None, "norm factor" : None},
                "r-music": {"fig" : None, "ax" : None},
                "esprit" : {"fig" : None, "ax" : None},
                "mvdr"   : {"fig" : None, "ax" : None, "norm factor" : None}}

        model_test_loss = evaluate_model(model, model_test_dataset, criterion=criterion,
                                        plot_spec= PLOT_SPECTRUM, figures=figures, model_type = model_type)
        print(f"{model_type} Test loss = {model_test_loss}")
        for algorithm in augmented_methods:
            hybrid_loss = evaluate_hybrid_model(model, model_test_dataset, samples_model,
                            criterion = hybrid_criterion, algorithm = algorithm, plot_spec= PLOT_SPECTRUM, figures = figures)
            print("hybrid {} test loss = {}".format(algorithm, hybrid_loss))
        for algorithm in subspace_methods:
            loss = evaluate_model_based(generic_test_dataset, samples_model, criterion=hybrid_criterion,
                                        plot_spec = PLOT_SPECTRUM, algorithm = algorithm, figures = figures)
            print("{} test loss = {}".format(algorithm.lower(), loss))
        # figures["mvdr"]["fig"].savefig(f"mvdr_spectrum_M_{M}_{mode}_T_{T}_SNR_{SNR}.pdf", bbox_inches='tight')
        # figures["music"]["fig"].savefig("{}_spectrum.pdf".format("music"), bbox_inches='tight')
    plt.show()
    print("end")