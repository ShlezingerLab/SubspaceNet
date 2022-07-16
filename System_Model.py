import numpy as np

class System_model(object):
    def __init__(self, scenario, N, M, DOA= None, freq_values = None):
        self.scenario = scenario                                    # Narrowband or Broadband
        self.N = N                                                  # Number of sensors in element
        self.M = M                                                  # Number of sources
        self.Scenario_define(freq_values)                           # Define parameters    
        self.Create_array()                                         # Define array indicies 
             
    def Scenario_define(self, freq_values):
        if self.scenario == "Broadband":
            ## frequencies initialization ##
            self.min_freq = freq_values[0]                          # Define minimal frequency value  
            self.max_freq = freq_values[1]                          # Define maximal frequency value  
            self.f_rng = np.linspace(start= self.min_freq,
                             stop= self.max_freq,
                              num= self.max_freq - self.min_freq,
                               endpoint = False)                    # Frequency range of interest  
            self.f_sampling = 2 * (self.max_freq)                   # Define sampling rate as twice the maximal frequency
            self.time_axis = np.linspace(0, 1, self.f_sampling,
                             endpoint = False)                      # Define time axis
            ## Array initialization ##
            self.dist = 1 / (2 * self.max_freq)                     # distance between array elements
        else: 
            ## frequencies initialization ##
            self.min_freq = None
            self.max_freq = None
            self.f_rng = None
            self.fs = None
            ## Array initialization ##
            self.dist = 1 / 2                                       # distance between array elements

    def Create_array(self):
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

#     # def NarrowBand_Signal_creation(self, S_mean, S_Var, SNR, mode):
#     #     f = 1
#     #     Amp = np.sqrt(1) * 20 * np.log10(SNR)
#     #     # Amp = (10 ** (SNR / 10))

#     #     if mode == "non-coherent": 
#     #         # create M non - coherent signals
#     #         return Amp * np.sqrt(S_Var) * (np.random.randn(self.M, self.T) + 1j * np.random.randn(self.M, self.T)) + S_mean
        
#     #     elif mode == "coherent": 
#     #         # create coherent signal such that all signals are the same
#     #         # and arrived from different angels
#     #         sig = Amp * np.sqrt(S_Var) * (np.random.randn(1, self.T) + 1j * np.random.randn(1, self.T)) + S_mean
#     #         return np.repeat(sig, self.M, axis = 0)

#     # def BroadBand_Signal_creation(self, SNR, mode, Carriers):
#     #     Amp = (np.sqrt(2) / 2 ) * 20 * np.log10(SNR) 
#     #     # Amp = (10 ** (SNR / 10))
#     #     sig = []
#     #     if mode == "non-coherent":
#     #         for f_c in Carriers:
#     #             Amp_f_c = Amp * (np.random.randn() + 1j * np.random.randn())
#     #             sig_fc = Amp_f_c * np.exp(2 * np.pi * 1j * f_c * self.time_axis)
#     #             sig.append(sig_fc)
#     #         # for Broadband scenario Noise represented in the frequency domain
#     #         return np.fft.fft(sig)
        
#     #     ###### should be updated ######
#     #     if mode == "OFDM": 
#     #         pass
#     #     if mode == "mod-OFDM": 
#     #         pass
#     #     else:
#     #         return 0
    
   