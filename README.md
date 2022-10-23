# DeepRootMUSIC
## Absract
Direction-of-arrival (DoA) estimation is a fundamental task in array processing.
A popular family of DoA estimation algorithms are subspace methods, which operate by dividing the measurements into distinct signal and noise subspaces.
Subspace methods, such as Root-MUSIC, require the sources to be non-coherent, and are considerably degraded when this does not hold.
In this work we propose Deep Root-MUSIC; a data-driven DoA estimator which augments Root-MUSIC with a deep neural network applied to the empirical 
autocorrelation of the input.
Deep Root-MUSIC learns how to divide the observations into distinguishable subspaces, thus leveraging data to cope with coherent sources, low SNR and limited
snapshots, while preserving the interpretability and the suitability of the model-based algorithm.
