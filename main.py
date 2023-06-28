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
from src.training import *
from src.evaluation import evaluate_augmented_model, evaluate_model_based
from src.evaluation import evaluate_model
# from src.utils import * 
from pathlib import Path

# Initialization
warnings.simplefilter("ignore")
os.system('cls||clear')
plt.close('all')

if __name__ == "__main__":
    # Initialize paths
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
                "LOAD_DATA"     : True,     # Loading data from dataset 
                "TRAIN_MODEL"   : True,     # Applying training operation
                "LOAD_MODEL"    : True,    # Load specific model
                "SAVE_MODEL"    : False,    # Saving tuned model
                "EVALUATE_MODE" : True}     # Evaluating desired algorithms
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = simulations_path / "results" / Path("scores" + dt_string_for_save + ".txt")
        sys.stdout = open(file_path, "w")
    # TODO: replace this section with system_model init
    # Define DNN model
    model_type = "SubspaceNet" # Set model type, options: "DA-MUSIC", "DeepCNN"  
    tau = 8 # Number of lags, relevant for "SubspaceNet" model
    ## Define system model parameters
    N = 8       # Number of sensors
    M = 2       # number of sources
    T = 100     # Number of observations, ideal >= 200
    SNR = 10    # Signal to noise ratio, ideal = 10
    ## Define signal parameters
    scenario = "NarrowBand" # signals type, options: "NarrowBand", "Broadband".
    mode = "coherent"   # signals nature, options: "non-coherent", "coherent"
    ## Array calibration values
    eta = 0                 # Deviation from sensor location, normalized by wavelength, ideal = 0
    geo_noise_var = 0       # Added noise for sensors response
    ## Simulation parameters
    samples_size = 100      # Overall dateset size
    train_test_ratio = 1  # training and testing datasets ratio 
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
        # Define which datasets to generate
        create_training_data = True # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset 
            train_dataset, _, _ = create_dataset(scenario= scenario, mode= mode, N= N, M= M , T= T,\
                samples_size = samples_size, tau = tau, model_type = model_type,\
                Save = True, datasets_path = datasets_path, true_doa = None, SNR = SNR,\
                eta=eta, geo_noise_var = geo_noise_var, phase = "train")
        if create_testing_data:
            # Generate test dataset
            test_dataset, generic_test_dataset, samples_model = create_dataset(scenario = scenario, mode = mode,\
                N= N, M= M , T= T, samples_size = int(train_test_ratio * samples_size), tau = tau,\
                model_type = model_type, Save = True, datasets_path = datasets_path,\
                true_doa = None, SNR = SNR, eta = eta, geo_noise_var = geo_noise_var, phase = "test")
    # Datasets loading
    if commands["LOAD_DATA"]:
        train_dataset, test_dataset, generic_test_dataset, samples_model = load_datasets(
            model_type, scenario, mode, samples_size, M, N, T, SNR, eta, geo_noise_var,\
            datasets_path=datasets_path, is_training = True)

    # Training stage
    if commands["TRAIN_MODEL"]:
        # Assign the training parameters object
        training_parameters = TrainingParams(model_type=model_type)\
            .set_batch_size(batch_size=256).set_epochs(epochs=10)\
            .set_model(system_model=samples_model, tau=tau)\
            .set_optimizer(optimizer="Adam", learning_rate=0.00001, weight_decay=1e-9)\
            .set_training_dataset(train_dataset=train_dataset)\
            .set_schedular(step_size=100, gamma=0.9)\
            .set_criterion()
        if commands["LOAD_MODEL"]:
            training_parameters.load_model(loading_path=saving_path / "final_models" /simulation_filename)
        # Print training simulation details
        simulation_summary(model_type=model_type, M=M, N=N, T=T, SNR=SNR,\
            scenario=scenario, mode=mode, eta=eta, geo_noise_var=geo_noise_var,\
            training_parameters = training_parameters,\
            phase="training", tau=tau)
        # Perform simulation training and evaluation stages
        model, loss_train_list, loss_valid_list = train(training_parameters = training_parameters,
            model_name= simulation_filename, saving_path=saving_path)
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(model.state_dict(), saving_path / "final_models" / Path(simulation_filename))
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(simulations_path / "results" / "plots" / Path(dt_string_for_save + r".png"))
        else:
            plt.show()
    
    # Evaluation stage
    if commands["EVALUATE_MODE"]:
        # Initialize figures dict for plotting
        PLOT_SPECTRUM = True
        figures = {"music"  : {"fig" : None, "ax" : None, "norm factor" : None},
                   "r-music": {"fig" : None, "ax" : None},
                   "esprit" : {"fig" : None, "ax" : None},
                   "mvdr"   : {"fig" : None, "ax" : None, "norm factor" : None}}
        # Select subspace method to augment via SubspaceNet
        augmented_methods = []
        if model_type.startswith("SubspaceNet"):
            augmented_methods = ["music", "mvdr", "esprit"]
        # Select subspace methods for comparison
        subspace_methods = ["esprit", "music", "r-music",
                            "mvdr", "sps-r-music", "sps-esprit",
                            "sps-music","bb-music"]
        # Define loss measure for evaluation
        loss_measure = "rmse"
        print(f"Loss measure = {loss_measure}")
        if loss_measure.startswith("rmse"):
            criterion = RMSPELoss() # define loss criterion
            subspace_criterion = RMSPE
        elif loss_measure.startswith("mse"):
            criterion = MSPELoss() # define loss criterion
            subspace_criterion = MSPE
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                model_type, scenario, mode, samples_size, M, N, T, SNR, eta,\
                geo_noise_var, datasets_path=datasets_path)
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(test_dataset,\
            batch_size=1, shuffle=False, drop_last=False)
        generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,\
            batch_size=1, shuffle=False, drop_last=False)
        # Load pre-trained model 
        if not commands["TRAIN_MODEL"]:
            # Define an evaluation parameters instance
            evaluation_params = TrainingParams()\
                .set_model(system_model=samples_model)\
                .load_model(loading_path=saving_path / Path(simulation_filename))
            model = evaluation_params.model
        # print simulation summary details
        simulation_summary(model_type=model_type, M=M, N=N, T=T, SNR=SNR, scenario=scenario,\
            mode=mode, eta=eta, geo_noise_var=geo_noise_var, phase="evaluation")
        # Evaluate SubspaceNet + Root-MUSIC algorithm performnces
        model_test_loss = evaluate_model(model=model, dataset=model_test_dataset,
            criterion=criterion, plot_spec= PLOT_SPECTRUM, figures=figures,
            model_type = model_type)
        print(f"{model_type} Test loss = {model_test_loss}")
        # Evaluate SubspaceNet augmented methods
        for algorithm in augmented_methods:
            hybrid_loss = evaluate_augmented_model(model=model, dataset=model_test_dataset,\
                system_model=samples_model, criterion = subspace_criterion, algorithm = algorithm,\
                plot_spec= True, figures = figures)
            print("hybrid {} test loss = {}".format(algorithm, hybrid_loss))
        # Evaluate classical subspace methods
        for algorithm in subspace_methods:
            loss = evaluate_model_based(generic_test_dataset, samples_model, 
                criterion=subspace_criterion, plot_spec = True, algorithm = algorithm,\
                figures = figures)
            print("{} test loss = {}".format(algorithm.lower(), loss))
    plt.show()
    print("end")