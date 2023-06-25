"""Subspace-Net 
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
import numpy as np
from src.system_model import SystemModel
from src.utils import D2R

class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        T (int): Number of observations.
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(mode: str, N_mean: float = 0, N_Var: float = 1, S_mean: float = 0, 
                         S_Var: float = 1, SNR: float = 10, eta: float = 0, 
                         geo_noise_var: float = 0): Creates samples based on the specified mode and parameters.
        noise_creation(N_mean, N_Var): Creates noise based on the specified mean and variance.
        signal_creation(mode: str, S_mean=0, S_Var=1, SNR=10): Creates signals based on the specified mode and parameters.
    """
    
    def __init__(self, scenario:str , N:int, M:int, observations:int, freq_values:list = [0, 500]):
        """Initializes a Samples object.

        Args:
            scenario (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            observations (int): Number of observations.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.

        """
        super().__init__(scenario, N, M, freq_values)
        self.T = observations
    
    def set_doa(self, doa):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """
        def create_doa_with_gap(M: int, gap: float):
            """ Create angles with a value gap.

            Args:
            -----
                M (int): Number of sources.
                gap (float): Minimal gap value.

            Returns:
            --------
                np.ndarray: DOA array.

            """
            while True:
                DOA = (np.round(np.random.rand(M) *  180 ,decimals = 2) - 90)
                DOA.sort()
                diff_angles = np.array([np.abs(DOA[i+1] - DOA[i]) for i in range(M-1)])
                if((np.sum(diff_angles > gap) == M - 1) and (np.sum(diff_angles < (180 - gap)) == M - 1)):
                    break
            return DOA        
        
        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.doa = np.array(create_doa_with_gap(M = self.M, gap = 15)) * D2R
        else:
            # Generate  
            self.doa = np.array(doa)                             
        
    def samples_creation(self, mode: str, N_mean:float=0, N_Var:float= 1,
                         S_mean:float= 0, S_Var:float= 1, SNR:float= 10,
                         eta:float = 0, geo_noise_var = 0):
        """ Creates samples based on the specified mode and parameters.

        Args:
        -----
            mode (str): Mode of signal creation. Options: "non-coherent", "coherent"
            N_mean (float, optional): Mean of the noise. Defaults to 0.
            N_Var (float, optional): Variance of the noise. Defaults to 1.
            S_mean (float, optional): Mean of the signal. Defaults to 0.
            S_Var (float, optional): Variance of the signal. Defaults to 1.
            SNR (float, optional): Signal-to-noise ratio. Defaults to 10.
            eta (float, optional): Noise correlation coefficient for distance from sensors nominal
                                   spacing.Defaults to 0.
            geo_noise_var (float, optional): Geometric noise variance. Defaults to 0.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the scenario is not defined.

        """
        signal = self.signal_creation(mode, S_mean, S_Var, SNR)
        noise = self.noise_creation(N_mean, N_Var)
        
        if self.scenario.startswith("NarrowBand"):
            A = np.array([self.steering_vec(theta, eta=eta, geo_noise_var=geo_noise_var) for theta in self.doa]).T
            samples = (A @ signal) + noise
            return samples, signal, A, noise

        elif self.scenario.startswith("Broadband"):
            samples = []
            SV = []
            
            for idx in range(self.f_sampling["Broadband"]):
                
                # mapping from index i to frequency f
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = - int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, :self.T]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(f"scenario {self.scenario} is not defined")

    def noise_creation(self, N_mean, N_Var):
        """ Creates noise based on the specified mean and variance.

        Args:
        -----
            N_mean (float): Mean of the noise.
            N_Var (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for NarrowBand scenario Noise represented in the time domain
        if self.scenario.startswith("NarrowBand"):
            return np.sqrt(N_Var) * (np.sqrt(2) / 2) * (np.random.randn(self.N, self.T)\
                            + 1j * np.random.randn(self.N, self.T)) + N_mean
        
        # for Broadband scenario Noise represented in the frequency domain
        elif self.scenario.startswith("Broadband"):
            noise = np.sqrt(N_Var) * (np.sqrt(2) / 2) * (np.random.randn(self.N, len(self.time_axis["Broadband"]))\
                            + 1j * np.random.randn(self.N, len(self.time_axis["Broadband"]))) + N_mean
            return np.fft.fft(noise)
    
    def signal_creation(self, mode:str, S_mean = 0, S_Var = 1, SNR = 10):
        """
        Creates signals based on the specified mode and parameters.

        Args:
        -----
            mode (str): Mode of signal creation.
            S_mean (float, optional): Mean of the signal. Defaults to 0.
            S_Var (float, optional): Variance of the signal. Defaults to 1.
            SNR (float, optional): Signal-to-noise ratio. Defaults to 10.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the scenario is not defined.
        """
        amplitude = (10 ** (SNR / 10))
        # NarrowBand signal creation 
        if self.scenario == "NarrowBand":
            if mode == "non-coherent": 
                # create M non-coherent signals
                return amplitude * (np.sqrt(2) / 2) * np.sqrt(S_Var) * (np.random.randn(self.M, self.T)\
                                + 1j * np.random.randn(self.M, self.T)) + S_mean
        
            elif mode == "coherent": 
                # Coherent signals: same amplitude and phase for all signals 
                sig = amplitude * (np.sqrt(2) / 2) * np.sqrt(S_Var) * (np.random.randn(1, self.T)\
                                + 1j * np.random.randn(1, self.T)) + S_mean
                return np.repeat(sig, self.M, axis = 0)
        
        # OFDM Broadband signal creation
        elif self.scenario.startswith("Broadband"):
            num_sub_carriers = self.max_freq["Broadband"]   # number of subcarriers per signal
            # create M non-coherent signals
            signal = np.zeros((self.M, len(self.time_axis["Broadband"]))) + 1j * np.zeros((self.M, len(self.time_axis["Broadband"])))
            if mode == "non-coherent":
                for i in range(self.M):
                    for j in range(num_sub_carriers):
                        sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                        signal[i] += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * self.time_axis["Broadband"] / num_sub_carriers)
                    signal[i] *=  (1/num_sub_carriers)          
                return np.fft.fft(signal)
             
            # Coherent signals: same amplitude and phase for all signals 
            signal = np.zeros((1, len(self.time_axis["Broadband"]))) + 1j * np.zeros((1, len(self.time_axis["Broadband"])))
            if mode == "coherent":
                for j in range(num_sub_carriers):
                    sig_amp = amplitude * (np.sqrt(2) / 2) * (np.random.randn(1) + 1j * np.random.randn(1))
                    signal += sig_amp * np.exp(1j * 2 * np.pi * j * len(self.f_rng["Broadband"]) * self.time_axis["Broadband"] / num_sub_carriers)
                signal *=  (1/num_sub_carriers)
                return np.tile(np.fft.fft(signal), (self.M, 1))
                
        else:
            raise Exception(f"scenario {self.scenario} is not defined")
