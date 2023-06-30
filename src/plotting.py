"""
Subspace-Net

Details
----------
Name: plotting.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 29/06/23

Purpose
----------
This module provides functions for plotting subspace methods spectrums,
like and RootMUSIC, MUSIC, and also beam patterns of MVDR.
 
Functions:
----------

plot_spectrum(predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
    spectrum: np.ndarray =None, roots: np.ndarray =None, algorithm:str ="music",
    figures:dict = None): Wrapper spectrum plotter based on the algorithm.
plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    Plot the MUSIC spectrum.
plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
    true_DOA: np.ndarray, algorithm: str): Plot the Root-MUSIC spectrum.
plot_mvdr_spectrum(system_model, figures: dict, spectrum: np.ndarray,
    true_DOA: np.ndarray, algorithm: str): Plot the MVDR spectrum.
initialize_figures(void): Generates template dictionary containing figure objects for plotting multiple spectrums.


"""
# Imports
from matplotlib import pyplot as plt
import numpy as np
import torch
from src.methods import MUSIC, RootMUSIC, MVDR
from src.utils import R2D

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
    
def initialize_figures():
  """Generates template dictionary containing figure objects for plotting multiple spectrums.

  Returns:
      (dict): The figures dictionary
  """  
  figures = {"music"  : {"fig" : None, "ax" : None, "norm factor" : None},
            "r-music": {"fig" : None, "ax" : None},
            "esprit" : {"fig" : None, "ax" : None},
            "mvdr"   : {"fig" : None, "ax" : None, "norm factor" : None}}
  return figures