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
from dataclasses import dataclass


@dataclass
class SystemModelParams:
    """Class for setting parameters of a system model.
    Initialize the SystemModelParams object.

    Parameters:
        None

    Attributes:
        M (int): Number of sources.
        N (int): Number of sensors.
        T (int): Number of observations.
        signal_type (str): Signal type ("NarrowBand" or "Broadband").
        freq_values (list): Frequency values for Broadband signal.
        signal_nature (str): Signal nature ("non-coherent" or "coherent").
        snr (float): Signal-to-noise ratio.
        eta (float): Level of deviation from sensor location.
        bias (float): Sensors locations bias deviation.
        sv_noise_var (float): Steering vector added noise variance.

    Returns:
        None
    """

    M = None  # Number of sources
    N = None  # Number of sensors
    T = None  # Number of observations
    signal_type = "NarrowBand"  # Signal type ("NarrowBand" or "Broadband")
    freq_values = [0, 500]  # Frequency values for Broadband signal
    signal_nature = "non-coherent"  # Signal nature ("non-coherent" or "coherent")
    snr = 10  # Signal-to-noise ratio
    eta = 0  # Sensor location deviation
    bias = 0  # Sensor bias deviation
    sv_noise_var = 0  # Steering vector added noise variance

    def set_parameter(self, name: str, value):
        """
        Set the value of the desired system model parameter.

        Args:
            name(str): the name of the SystemModelParams attribute.
            value (int, float, optional): the desired value to assign.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.__setattr__(name, value)
        return self


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
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
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()
        # Define array indices
        self.create_array()

    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                start=self.min_freq["Broadband"],
                stop=self.max_freq["Broadband"],
                num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            "NarrowBand": None,
            "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]),
        }
        # Define time axis
        self.time_axis = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                0, 1, self.f_sampling["Broadband"], endpoint=False
            ),
        }
        # distance between array elements
        self.dist = {
            "NarrowBand": 1 / 2,
            "Broadband": 1
            / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])),
        }

    def create_array(self):
        """create an array of sensors locations"""
        self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)

    def steering_vec(
        self, theta: np.ndarray, f: float = 1, array_form="ULA", nominal=False
    ):
        """Computes the steering vector based on the specified parameters.

        Args:
        -----
            theta (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            array_form (str, optional): Array form. Defaults to "ULA".
            nominal (bool): flag for creating sv without array mismatches.

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        f_sv = {"NarrowBand": 1, "Broadband": f}
        if array_form.startswith("ULA"):
            # define uniform deviation in spacing (for each sensor)
            if not nominal:
                # Calculate uniform bias for sensors locations
                uniform_bias = np.random.uniform(
                    low=-1 * self.params.bias, high=self.params.bias, size=1
                )
                # Calculate non-uniform bias for each pair of sensors
                mis_distance = np.random.uniform(
                    low=-1 * self.params.eta, high=self.params.eta, size=self.params.N
                )
                # Calculate additional steering vector noise
                mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * (
                    np.random.randn(self.params.N)
                )
            # If calculation is applied through method (array mismatches are not known).
            else:
                mis_distance, mis_geometry_noise, uniform_bias = 0, 0, 0

            return (
                np.exp(
                    -2
                    * 1j
                    * np.pi
                    * f_sv[self.params.signal_type]
                    * (uniform_bias + mis_distance + self.dist[self.params.signal_type])
                    * self.array
                    * np.sin(theta)
                )
                + mis_geometry_noise
            )
        else:
            raise Exception(
                f"SystemModel.steering_vec: array form {array_form} is not defined"
            )

    def __str__(self):
        """Returns a string representation of the SystemModel object.
        ...

        """
        print("System Model Summery:")
        for key, value in self.__dict__.items():
            print(key, " = ", value)
        return "End of Model"
