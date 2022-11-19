import numpy as np

class System_model(object):
    def __init__(self, scenario:str , N:int, M:int, freq_values = None):
        self.scenario = scenario                                    # Narrowband or Broadband
        self.N = N                                                  # Number of sensors in element
        self.M = M                                                  # Number of sources
        self.scenario_define(freq_values)                           # Define parameters    
        self.create_array()                                         # Define array indicies 
             
    def scenario_define(self, freq_values):
        if self.scenario.startswith("Broadband"):
            ## frequencies initialization ##
            self.min_freq = freq_values[0]                          # Define minimal frequency value  
            self.max_freq = freq_values[1]                          # Define maximal frequency value  
            self.f_rng = np.linspace(start=self.min_freq, stop=self.max_freq,
                                     num=self.max_freq - self.min_freq,
                                     endpoint = False)                    # Frequency range of interest  
            self.f_sampling = 2 * (self.max_freq)                   # Define sampling rate as twice the maximal frequency
            # self.f_sampling = 2000                                    # Define sampling rate as twice the maximal frequency
            self.time_axis = np.linspace(0, 1, self.f_sampling, endpoint = False)                      # Define time axis
            
            ## Array initialization ##
            self.dist = 1 / (2 * self.max_freq)                     # distance between array elements
            # self.dist = 1 / (self.max_freq - self.min_freq)           # distance between array elements
        
        elif self.scenario.startswith("NarrowBand"):
            ## frequencies initialization ##
            self.min_freq = None
            self.max_freq = None
            self.f_rng = None
            self.fs = None
            ## Array initialization ##
            self.dist = 1 / 2                                       # distance between array elements
        else:
            raise Exception("Scenario: {} is not defined".format(self.scenario))
    
    def create_array(self):
        self.array = np.linspace(0, self.N, self.N, endpoint = False)   # create array of sensors locations
    
    def SV_Creation(self, theta, f=1, Array_form= "ULA"):
        if self.scenario == "NarrowBand": f = 1
        if Array_form == "ULA":
            return np.exp(-2 * 1j * np.pi * f * self.dist * self.array * np.sin(theta))

    def __str__(self):
        print("System Model Summery:")
        for key,value in self.__dict__.items():
            print (key, " = " ,value)
        return "End of Model"