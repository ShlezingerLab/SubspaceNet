# SubspaceNet: Deep Learning-Aided Subspace Methods for DoA
This repository includes the source code used in our recent paper:

Dor H. Shmuel, Julian P. Merkofer, Guy Revach, Ruud J. G. van Sloun, Nir Shlezinger
"[SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation]([url](https://arxiv.org/abs/2306.02271))" 

## Absract
Direction of arrival (DoA) estimation is a fundamental task in array processing. A popular family of DoA estimation algorithms are subspace methods, which operate by dividing the measurements into distinct signal and noise subspaces. Subspace methods, such as MUSIC and Root-MUSIC, rely on several restrictive assumptions, including narrow-band non-coherent sources and fully calibrated arrays, and their performance is considerably degraded when these do not hold.
In this work we propose SubspaceNet; a data-driven DoA universal estimator which learns how to divide the observations into distinguishable subspaces. This is achieved by utilizing a dedicated deep neural network to learn the empirical autocorrelation of the input, by training it as part of the Root-MUSIC method, leveraging the inherent differentiability of this specific DoA estimator, while removing the need to provide a ground-truth decomposable autocorrelation matrix. Once trained, the resulting SubspaceNet serves as a universal surrogate covariance estimator that can be applied in combination with any subspace-based \ac{doa} estimation method, allowing its successful application in challenging setups. SubspaceNet is shown to enable various DoA estimation algorithms to cope with coherent sources, wideband signals, low SNR, array mismatches, and limited snapshots, while preserving the interpretability and the suitability of classic subspace methods. 

## Overview
This repository consists of following Python scripts:
* `main.py` this scripts is the interface for applying the proposed algorithms, by wrapping all the required procedures and parameters for the simulation.
* `src.simulation_handler.py` handling the simulation of the algorithms, including the training and evaluation stage.
* `src.criterions.py` defines document several loss functions (RMSPELoss and MSPELoss).
* `src.data_handler.py` handles the creation and processing of synthetic datasets based on specified parameters and model types.
* `src.methods.py` defines the classical and model-based methods, which used for simulation.
* `src.models.py` defines the tested NN-models and the model-based DL models, which used for simulation.
* `src.signal_creation.py` defines and creates signals and observations. Inherits from SystemModel class.
* `src.system_model.py` defines the SystemModel class for defining the settings of the DoA estimation system model.
* `src.utils.py` defines some helpful functions.

## Requirements
This script requires that requirements.txt will be installed within the Python environment you are running in.
