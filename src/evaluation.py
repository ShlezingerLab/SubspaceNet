"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
plot_spectrum: Wrapper spectrum plotter based on the algorithm.
plot_music_spectrum: Plot the MUSIC spectrum.
plot_root_music_spectrum: Plot the Root-MUSIC spectrum.
plot_mvdr_spectrum: Plot the MVDR spectrum.


"""
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, MSPELoss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *
from src.models import SubspaceNet

def evaluate_model(model, dataset: list, criterion: nn.Module,
        plot_spec: bool = False, figures: dict = None, model_type: str="SubspaceNet"):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """  

    # Initialize values
    overall_loss = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for data in dataset:
            X, DOA = data
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            model_output = model(X)
            if model_type.startswith("DA-MUSIC"):
              # Deep Augmented MUSIC
              DOA_predictions = model_output
            elif model_type.startswith("DeepCNN"):
                # Deep CNN
                if isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output
                    # find peaks in the pseudo spectrum of probabilities
                    DOA_predictions = get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
                    DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
                elif isinstance(criterion, [RMSPELoss, MSPELoss]):
                    # If evaluation performed over testset, loss is RMSPE / MSPE 
                    DOA_predictions = model_output
                else:
                    raise Exception(f"evaluate_model: Loss criterion is not defined for {model_type} model")
            elif model_type.startswith("SubspaceNet"):
                # Default - SubSpaceNet
                DOA_predictions = model_output[0]
            else:
                raise Exception(f"evaluate_model: Model type {model_type} is not defined")
            # Compute prediction loss
            if model_type.startswith("DeepCNN") and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            else:
                eval_loss = criterion(DOA_predictions, DOA)
            # add the batch evaluation loss to epoch loss              
            overall_loss += eval_loss.item()
        overall_loss = overall_loss / test_length
    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.startswith("SubspaceNet"):
        DOA_all = model_output[1]
        roots = model_output[2]
        plot_spectrum(predictions=DOA_all * R2D, true_DOA=DOA[0] * R2D,\
            roots=roots,algorithm="SubNet+R-MUSIC", figures=figures)
    return overall_loss

def evaluate_augmented_model(model: SubspaceNet, dataset, system_model, criterion = RMSPE,\
    algorithm:str = "music", plot_spec:bool = False, figures:dict = None):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = [] 
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {"mvdr"     : MVDR(system_model),
                "music"   : MUSIC(system_model),
                "esprit"  : Esprit(system_model)}
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
      raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
      for i, data in enumerate(dataset):
            X, DOA = data
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(X = X, mode="SubspaceNet", model=model)
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                predictions, M = method_output[0], method_output[-1]
                # If the amount of predictions is less than the amount of sources
                predictions = add_random_predictions(M, predictions, algorithm)
                # Calculate loss criterion
                loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(predictions=predictions, true_DOA=DOA * R2D,
                        system_model=system_model, spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper() , figures=figures)
    return np.mean(hybrid_loss)

def evaluate_model_based(dataset:list, system_model, criterion: RMSPE,
    plot_spec=False, algorithm:str="music", figures:dict=None):
  """
  Evaluate different model-based algorithms on a given dataset.

  Args:
      dataset (list): The evaluation dataset.
      system_model (SystemModel): The system model for the algorithms.
      criterion: The loss criterion for evaluation. Defaults to RMSPE.
      plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
      algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
      figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

  Returns:
      float: The average evaluation loss.

  Raises:
      Exception: If the algorithm is not supported.
  """
  # Initialize parameters for evaluation
  loss_list = []
  for i, data in enumerate(dataset):
    X, doa = data
    X = X[0]
    # Root-MUSIC algorithms
    if "r-music" in algorithm:
      root_music = RootMUSIC(system_model)
      if algorithm.startswith("sps"):
        # Spatial smoothing
        predictions, roots, M, predictions_all, _ = root_music.narrowband(X=X, mode="spatial_smoothing")
      else:
        # Conventional
        predictions, roots, M, predictions_all, _ = root_music.narrowband(X=X, mode="sample")
      # If the amount of predictions is less than the amount of sources
      predictions = add_random_predictions(M, predictions, algorithm)
      # Calculate loss criterion
      loss = criterion(predictions, doa * R2D)
      loss_list.append(loss)
      # Plot spectrum
      if plot_spec and i == len(dataset.dataset) - 1:
        plot_spectrum(predictions=predictions_all, true_DOA=doa[0] * R2D,
            roots=roots, algorithm=algorithm.upper(), figures= figures)
    # MUSIC algorithms
    elif "music" in algorithm:
      music = MUSIC(system_model)
      if algorithm.startswith("bb"):
        # Broadband MUSIC
        predictions, spectrum, M = music.broadband(X=X)
      elif algorithm.startswith("sps"):
        # Spatial smoothing
        predictions, spectrum, M = music.narrowband(X=X, mode="spatial_smoothing")
      elif algorithm.startswith("music"):
        # Conventional
        predictions, spectrum, M = music.narrowband(X=X, mode="sample")
      # If the amount of predictions is less than the amount of sources
      predictions = add_random_predictions(M, predictions, algorithm)
      # Calculate loss criterion
      loss = criterion(predictions, doa * R2D)
      loss_list.append(loss)
      # Plot spectrum
      if plot_spec and i == len(dataset.dataset) - 1:
        plot_spectrum(predictions=predictions, true_DOA=doa * R2D, system_model=system_model,\
          spectrum=spectrum, algorithm=algorithm.upper(), figures=figures)
    
    # ESPRIT algorithms
    elif "esprit" in algorithm:
      esprit = Esprit(system_model)
      if algorithm.startswith("sps"):
        # Spatial smoothing
        predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
      else:
        # Conventional
        predictions, M = esprit.narrowband(X=X, mode="sample")
      # If the amount of predictions is less than the amount of sources
      predictions = add_random_predictions(M, predictions, algorithm)      
      # Calculate loss criterion
      loss = criterion(predictions, doa * R2D)
      loss_list.append(loss)

    # MVDR algorithm
    elif algorithm.startswith("mvdr"):
      mvdr = MVDR(system_model)
      # Conventional
      _, spectrum = mvdr.narrowband(X=X, mode="sample")
      # Plot spectrum
      if plot_spec and i == len(dataset.dataset) - 1:
        plot_spectrum(predictions=None, true_DOA=doa * R2D, system_model=system_model,\
          spectrum=spectrum, algorithm=algorithm.upper(), figures= figures)
    else:
      raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")
  return np.mean(loss_list)

def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
      predictions = np.array(predictions)
    while(predictions.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(predictions, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)
    return predictions

def plot_spectrum(predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
    spectrum: np.ndarray =None, roots: np.ndarray =None, algorithm:str ="music",
    figures:dict = None):
  """
  Wrapper spectrum plotter based on the algorithm.

  Args:
      predictions (np.ndarray): The predicted DOA values.
      true_DOA (np.ndarray): The true DOA values.
      system_model: The system model.
      spectrum (np.ndarray): The spectrum values.
      roots (np.ndarray): The roots for Root-MUSIC algorithm.
      algorithm (str): The algorithm used.
      figures (dict): Dictionary containing figure objects for plotting.

  Raises:
      Exception: If the algorithm is not supported.

  """
  # Convert predictions to 1D array
  if isinstance(predictions, (np.ndarray, list, torch.Tensor)):
    predictions = np.squeeze(np.array(predictions))
  # Plot MUSIC spectrums
  if "music" in algorithm.lower() and not ("r-music" in algorithm.lower()):
    plot_music_spectrum(system_model, figures, spectrum, algorithm)
  elif "mvdr" in algorithm.lower():
    plot_mvdr_spectrum(system_model, figures, spectrum, true_DOA, algorithm)
  elif "r-music" in algorithm.lower():
    plot_root_music_spectrum(roots, predictions, true_DOA, algorithm)
  else:
    raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")

def plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    """
    Plot the MUSIC spectrum.

    Args:
        system_model (SystemModel): The system model.
        figures (dict): Dictionary containing figure objects for plotting.
        spectrum (np.ndarray): The spectrum values.
        algorithm (str): The algorithm used.

    """
    # Initialize MUSIC instance
    music = MUSIC(system_model)
    angels_grid = music._angels * R2D
    # Initialize plot for spectrum
    if figures["music"]["fig"] == None:
      plt.style.use('default')
      figures["music"]["fig"] = plt.figure(figsize=(8, 6))
      # plt.style.use('plot_style.txt')
    if figures["music"]["ax"] == None:
      figures["music"]["ax"] = figures["music"]["fig"].add_subplot(111)
    # Set labels titles and limits
    figures["music"]["ax"].set_xlabel("Angels [deg]")
    figures["music"]["ax"].set_ylabel("Amplitude")
    figures["music"]["ax"].set_ylim([0.0, 1.01])
    # Apply normalization factor for multiple plots
    figures["music"]["norm factor"] = None
    if figures["music"]["norm factor"] != None:
      # Plot music spectrum
      figures["music"]["ax"].plot(angels_grid , spectrum / figures["music"]["norm factor"], label=algorithm)
    else:
      # Plot normalized music spectrum
      figures["music"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    # Set legend
    figures["music"]["ax"].legend()

def plot_mvdr_spectrum(system_model, figures: dict, spectrum: np.ndarray,
    true_DOA: np.ndarray, algorithm: str):
    """
    Plot the MVDR spectrum.

    Args:
        system_model (SystemModel): The system model.
        figures (dict): Dictionary containing figure objects for plotting.
        spectrum (np.ndarray): The spectrum values.
        algorithm (str): The algorithm used.
        true_DOA (np.ndarray): The true DOA values.

    """
    # Initialize MVDR instance
    mvdr = MVDR(system_model)
    # Initialize plot for spectrum
    if figures["mvdr"]["fig"] == None:
      plt.style.use('default')
      figures["mvdr"]["fig"] = plt.figure(figsize=(8, 6))
    if figures["mvdr"]["ax"] == None:
      figures["mvdr"]["ax"] = figures["mvdr"]["fig"].add_subplot(111, polar=True)
    # Set axis location and limits
    figures["mvdr"]["ax"].set_theta_zero_location('N')
    figures["mvdr"]["ax"].set_theta_direction(-1)
    figures["mvdr"]["ax"].set_thetamin(-90)
    figures["mvdr"]["ax"].set_thetamax(90)
    figures["mvdr"]["ax"].set_ylim([0.0, 1.01])
    # Plot normalized mvdr beam pattern
    figures["mvdr"]["ax"].plot(mvdr._angels , spectrum / np.max(spectrum), label=algorithm)
    # marker in "x" true DoA's
    for doa in true_DOA[0]:
      figures["mvdr"]["ax"].plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    # Set leagend
    figures["mvdr"]["ax"].legend()

def plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
    true_DOA: np.ndarray, algorithm: str):
    """
    Plot the Root-MUSIC spectrum.

    Args:
        roots (np.ndarray): The roots for Root-MUSIC polynomyal.
        predictions (np.ndarray): The predicted DOA values.
        true_DOA (np.ndarray): The true DOA values.
        algorithm (str): The algorithm used.

    """
    # Initialize figure
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    # Set axis location and limits
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)
    # plot roots ang angles 
    for i in range(len(predictions)):
      angle = predictions[i]
      r = np.abs(roots[i])
      ax.set_ylim([0, 1.2])
      ax.set_yticks([0, 1])
      ax.plot([0, angle * np.pi / 180], [0, r], marker='o')
    for doa in true_DOA:
      ax.plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    ax.set_xlabel("Angels [deg]")
    ax.set_ylabel("Amplitude")
    plt.savefig("data/spectrums/{}_spectrum.pdf".format(algorithm), bbox_inches='tight')