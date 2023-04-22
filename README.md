# SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation
## Absract
Direction of arrival (DoA) estimation is a fundamental task in array processing.
A popular family of DoA estimation algorithms are subspace methods, which operate by dividing the measurements into distinct signal and noise subspaces.
Subspace methods, such as multiple signal classification (MUSIC) and Root-MUSIC, require the sources to be non-coherent, and are considerably degraded when this does not hold.
In this work we propose SubspaceNet; a data-driven DoA estimator which learns how to divide the observations into distinguishable subspaces. This is achieved by utilizing a dedicated deep neural network to learn the empirical autocorrelation of the input, while training it as part of the Root-MUSIC method, leveraging the inherent differentiability of this specific DoA estimator, while mitigating the need to provide a ground-truth decomposable autocorrelation matrix.
Once trained, the resulting Subspace Net (SubspaceNet) can be applied with different subspace DoA estimation methods, allowing them to be successfully applied in challenging setups.
The proposed approach is shown to successfully enable various DoA estimation algorithms to cope with coherent sources, low SNR, array mismatches, and limited snapshots, while preserving the interpretability and the suitability of classic subspace methods.
