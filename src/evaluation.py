# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, MSPELoss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR
from src.utils import *

def evaluate_model(model, dataset, criterion, plot_spec = False,\
    figures = None, model_type="SubspaceNet"):
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
    if plot_spec and model_type.startswith("SubspaceNet"):
        DOA_all = model_output[1]
        roots = model_output[2]
        plot_spectrum(DOA_prediction=DOA_all * R2D, true_DOA=DOA[0] * R2D,\
            roots=roots,algorithm="SubNet+R-MUSIC", figures=figures)
    return overall_loss

def evaluate_hybrid_model(model, dataset, system_model, criterion = RMSPE,\
    algorithm = "music", plot_spec = False, figures = None):
    # Initialize parameters for evaluation
    hybrid_loss = [] 
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():   
        for i, data in enumerate(dataset):
            X, DOA = data
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # SubspaceNet + MVDR
            if algorithm.startswith("mvdr"):
                # Create mvdr method instance
                mvdr = MVDR(system_model)
                # Calculate mvdr with SubspaceNet augmentation
                MVDR_spectrum = mvdr.narrowband(X = X, mode="SubspaceNet", model=model)
                if plot_spec and i == len(dataset.dataset) - 1:
                    figures["mvdr"]["norm factor"] = np.max(MVDR_spectrum)
                    plot_spectrum(DOA_prediction=None, true_DOA=DOA * R2D, system_model=system_model,\
                            spectrum=MVDR_spectrum, algorithm="SubNet+MVDR", figures=figures)
                hybrid_loss = 0
            else:
                # SubspaceNet + MUSIC
                if algorithm.startswith("music"):
                    # Create music method instance
                    music = MUSIC(system_model)
                    # Calculate music with SubspaceNet augmentation
                    predictions, spectrum, M = music.narrowband(X = X, mode="SubspaceNet", model=model)
                    # Calculate doa predictions
                    predictions = music._angels[predictions] * R2D
                    # Take the first M predictions
                    predictions = predictions[:M][::-1]
                # SubspaceNet + ESPRIT
                elif algorithm.startswith("esprit"):
                # Create music method instance
                    esprit = Esprit(system_model)
                    # Calculate esprit with SubspaceNet augmentation
                    predictions, M = esprit.narrowband(X=X, mode="SubspaceNet", model=model)
                # If the amount of predictions is less than the amount of sources
                while(predictions.shape[0] < M):
                    print("Cant estimate M sources - hybrid {}".format(algorithm))
                    predictions = np.insert(predictions, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)
                # Calculate loss criterion
                loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
                if plot_spec and i == len(dataset.dataset) - 1:
                    # Todo (Bug here)
                    figures["music"]["norm factor"] = np.max(spectrum)
                    plot_spectrum(DOA_prediction=predictions, true_DOA=DOA * R2D, system_model=system_model,\
                                spectrum=spectrum, algorithm="SubNet+MUSIC", figures=figures)
    return np.mean(hybrid_loss)

def evaluate_model_based(dataset_mb, system_model, criterion=RMSPE, plot_spec=False, algorithm="music", figures=None):
  loss_list = []
  for i, data in enumerate(dataset_mb):
    X, doa = data
    X = X[0]
    # TODO: unify BB music and music under the same method
    # Root-MUSIC algorithms
    if "r-music" in algorithm:
      root_music = RootMUSIC(system_model)
      if algorithm.startswith("sps"):
        DOA_pred, roots, M, DOA_pred_all, _ = root_music.narrowband(X=X, mode="spatial_smoothing")
      else:
        DOA_pred, roots, M, DOA_pred_all, _ = root_music.narrowband(X=X, mode="sample")
      # if algorithm cant estimate M sources, randomize angels
      while(DOA_pred.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        DOA_pred = np.insert(DOA_pred, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)        
      loss = criterion(DOA_pred, doa * R2D)
      loss_list.append(loss)
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=DOA_pred_all, true_DOA=doa[0] * R2D, roots=roots, algorithm=algorithm.upper(), figures= figures)
    # MUSIC algorithms
    elif "music" in algorithm:
      music = MUSIC(system_model)
      if algorithm.startswith("bb"):
        DOA_pred, spectrum, M = music.broadband(X=X)
      elif algorithm.startswith("sps"):
        DOA_pred, spectrum, M = music.narrowband(X=X, mode="spatial_smoothing")
      elif algorithm.startswith("music"):
        DOA_pred, spectrum, M = music.narrowband(X=X, mode="sample")
      
      DOA_pred = music._angels[DOA_pred] * R2D  # Convert from radians to degrees
      predicted_DOA = DOA_pred[:M][::-1]  # Take First M predictions
      # if algorithm cant estimate M sources, randomize angels
      while(predicted_DOA.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        predicted_DOA = np.insert(predicted_DOA, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)
      loss = criterion(predicted_DOA, doa * R2D)
      loss_list.append(loss)
      # plot spectrum
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=predicted_DOA, true_DOA=doa * R2D, system_model=system_model,\
          spectrum=spectrum, algorithm=algorithm.upper(), figures=figures)
    
    ### ESPRIT algorithms ###
    elif "esprit" in algorithm:
      esprit = Esprit(system_model)
      if algorithm.startswith("sps"):
        DOA_pred, M = esprit.narrowband(X=X, mode="spatial_smoothing")
      else:
        DOA_pred, M = esprit.narrowband(X=X, mode="sample")
      # if algorithm cant estimate M sources, randomize angels
      while(DOA_pred.shape[0] < M):
        print(f"{algorithm}: cant estimate M sources")
        DOA_pred = np.insert(DOA_pred, 0, np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00)        
      loss = criterion(DOA_pred, doa * R2D)
      loss_list.append(loss)

    # MVDR evaluation
    elif algorithm.startswith("mvdr"):
      mvdr = MVDR(system_model)
      spectrum = mvdr.narrowband(X=X, mode="sample")
      if plot_spec and i == len(dataset_mb.dataset) - 1:
        plot_spectrum(DOA_prediction=None, true_DOA=doa * R2D, system_model=system_model,\
          spectrum=spectrum, algorithm=algorithm.upper(), figures= figures)
  return np.mean(loss_list)


def plot_spectrum(DOA_prediction, true_DOA, system_model=None, spectrum=None, roots=None,
        algorithm="music", figures = None):
  if isinstance(DOA_prediction, (np.ndarray, list, torch.Tensor)):
    DOA_prediction = np.squeeze(np.array(DOA_prediction))
  # MUSIC algorithms
  if "music" in algorithm.lower() and not ("r-music" in algorithm.lower()):
    if figures["music"]["fig"] == None:
      plt.style.use('default')
      figures["music"]["fig"] = plt.figure(figsize=(8, 6))
      # plt.style.use('plot_style.txt')
    if figures["music"]["ax"] == None:
      figures["music"]["ax"] = figures["music"]["fig"].add_subplot(111)

    music = MUSIC(system_model)
    angels_grid = music._angels * R2D
    # ax.set_title(algorithm.upper() + "spectrum")
    figures["music"]["ax"].set_xlabel("Angels [deg]")
    figures["music"]["ax"].set_ylabel("Amplitude")
    figures["music"]["ax"].set_ylim([0.0, 1.01])
    figures["music"]["norm factor"] = None
    if figures["music"]["norm factor"] != None:
      figures["music"]["ax"].plot(angels_grid , spectrum / figures["music"]["norm factor"], label=algorithm)
    else:
      figures["music"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    figures["music"]["ax"].legend()

  elif "mvdr" in algorithm.lower():
    mvdr = MVDR(system_model)
    if figures["mvdr"]["fig"] == None:
      plt.style.use('default')
      figures["mvdr"]["fig"] = plt.figure(figsize=(8, 6))
      # plt.style.use('plot_style.txt')
    if figures["mvdr"]["ax"] == None:
      figures["mvdr"]["ax"] = figures["mvdr"]["fig"].add_subplot(111, polar=True)
    # mb.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 3600, endpoint=False)
    # ax.set_xlabel("Angels [deg]")
    figures["mvdr"]["ax"].set_theta_zero_location('N')
    figures["mvdr"]["ax"].set_theta_direction(-1)
    figures["mvdr"]["ax"].set_thetamin(-90)
    figures["mvdr"]["ax"].set_thetamax(90)
    
    angels_grid = mvdr._angels
    # figures["mvdr"]["ax"].set_title(algorithm.upper() + "spectrum")
    # ax.set_xlabel("Angels [deg]")
    # ax.set_ylabel("Amplitude")
    figures["mvdr"]["ax"].set_ylim([0.0, 1.01])
    figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm)
    for doa in true_DOA[0]:
      figures["mvdr"]["ax"].plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum), label=algorithm + " pattern")
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum / np.max(spectrum),  label=algorithm.upper() + " pattern")
      # figures["mvdr"]["ax"].plot(angels_grid , spectrum, label=algorithm + " pattern")
    figures["mvdr"]["ax"].legend()
  
  elif "r-music" in algorithm.lower():
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    # plt.style.use('plot_style.txt')
    ax = fig.add_subplot(111, polar=True)

    # ax.set_xlabel("Angels [deg]")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)

    for i in range(len(DOA_prediction)):
      angle = DOA_prediction[i]
      r = np.abs(roots[i])
      ax.set_ylim([0, 1.2])
      # ax.set_yticks([0, 1, 2, 3, 4])
      ax.set_yticks([0, 1])
      ax.plot([0, angle * np.pi / 180], [0, r], marker='o')
    for doa in true_DOA:
      ax.plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    # ax.set_xlabel("Angels [deg]")
    # ax.set_ylabel("Amplitude")
    plt.savefig("data/spectrums/{}_spectrum.pdf".format(algorithm), bbox_inches='tight')