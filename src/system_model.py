"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np

class SystemModel(object):
    def __init__(self, scenario:str , N:int, M:int, freq_values:list = None):
        
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            scenario (str): Signals type. Options: "NarrowBand", "Broadband_OFDM", "Broadband_simple".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the scenario parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                         eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.scenario = scenario
        self.N = N
        self.M = M
        self.define_scenario_params(freq_values)       # Assign scenario parameters    
        self.create_array()                     # Define array indices 
             
    def define_scenario_params(self, freq_values:list):
        """Defines the scenario parameters based on the specified frequency values.

        Args:
        -----
            freq_values (list): Frequency range for broadband signals.

        """
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None,
                         "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None,
                         "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {"NarrowBand": None,
                      "Broadband": np.linspace(start=self.min_freq["Broadband"], stop=self.max_freq["Broadband"],\
                       num=self.max_freq["Broadband"] - self.min_freq["Broadband"], endpoint = False)}
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {"NarrowBand": None,
                           "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])}
        # Define time axis   
        self.time_axis = {"NarrowBand": None,
                          "Broadband": np.linspace(0, 1, self.f_sampling["Broadband"], endpoint = False)}
        # distance between array elements
        self.dist = {"NarrowBand": 1 / 2,
                     "Broadband": 1 / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]))}
        
    def create_array(self):
        """ create an array of sensors locations
        """        
        self.array = np.linspace(0, self.N, self.N, endpoint = False)   
    
    def steering_vec(self, theta:np.ndarray, f:float=1, array_form= "ULA",
                     eta:float = 0, geo_noise_var:float = 0):
        """Computes the steering vector based on the specified parameters.

        Args:
        -----
            theta (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".
            eta (float, optional): Sensor distance deviation. Defaults to 0.
            geo_noise_var (float, optional): Steering vector noise variance. Defaults to 0.

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        f_sv = {"NarrowBand": 1, "Broadband": f}
        if array_form.startswith("ULA"):
            # define uniform deviation in spacing (for each sensor)
            mis_distance = np.random.uniform(low= -1 * eta, high= eta, size=self.N)
            # define noise added to steering vector
            mis_geometry_noise = np.sqrt(geo_noise_var) * (np.random.randn(self.N))
            return np.exp(-2 * 1j * np.pi * f_sv[self.scenario]\
                    * (mis_distance + self.dist[self.scenario]) * self.array\
                    * np.sin(theta)) + mis_geometry_noise
            
    def __str__(self):
        """Returns a string representation of the SystemModel object.
        ...

        """
        print("System Model Summery:")
        for key,value in self.__dict__.items():
            print (key, " = " ,value)
        return "End of Model"