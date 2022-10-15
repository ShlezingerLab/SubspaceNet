import numpy as np
from System_Model import *

def create_DOA_with_gap(M, gap):
    while(True):
        DOA = np.round(np.random.rand(M) *  180 ,decimals = 2) - 90.00
        DOA.sort()
        difference_between_angles = np.array([np.abs(DOA[i+1] - DOA[i]) for i in range(M-1)])
        if(np.sum(difference_between_angles > gap) == M - 1 and np.sum(difference_between_angles < (180 - gap)) == M - 1):
            break
    return DOA

def create_closely_spaced_DOA(M, gap):
    if (M == 2):
        first_DOA = np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00
        second_DOA = ((first_DOA + gap + 90 ) % 180) - 90
        return np.array([first_DOA, second_DOA])
    DOA = [np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00]
    while(len(DOA) < M):
        candidate_DOA = np.round(np.random.rand(1) *  180 ,decimals = 2) - 90.00
        difference_between_angles = np.array([np.abs(candidate_DOA - DOA[i]) for i in range(len(DOA))])
        if(np.sum(difference_between_angles < gap) == len(DOA) or np.sum((180 - difference_between_angles) < gap) == len(DOA)):
            DOA.append(candidate_DOA)
    return np.array(DOA)

class Sampels(object):
    def __init__(self, System_model, DOA, observations):
        self.scenario = System_model.scenario
        self.N = System_model.N
        self.M = System_model.M
        self.T = observations
        self.SV_Creation = System_model.SV_Creation
        if DOA == None:
          # self.DOA = np.array(np.pi * (np.random.rand(self.M) - 0.5))         # generate aribitrary DOA angels
        #   self.DOA = (np.pi / 180) * np.array(create_DOA_with_gap(M = self.M, gap = 15)) # (~0.2 rad)
          self.DOA = (np.pi / 180) * np.array(create_closely_spaced_DOA(M = self.M, gap = 10)) # (~0.2 rad)
            # self.DOA = np.array(np.round((np.pi * ((np.random.rand(self.M) - 0.5))),decimals=2))
        else: 
          self.DOA = (np.pi / 180) * np.array(DOA)                              # define DOA angels
    
    def Sampels_creation(self, mode, N_mean= 0, N_Var= 1, S_mean= 0, S_Var= 1, SNR= 10, Carriers= None):
        '''
        @mode = represent the specific mode in the specific scenario
                e.g. "Broadband" scenario in "non-coherent" mode
        '''
        signal = self.Signal_creation(mode, S_mean, S_Var, SNR, Carriers)
        noise = self.Noise_Creation(N_mean, N_Var)
        A = np.array([self.SV_Creation(theta) for theta in self.DOA]).T
        
        if self.scenario == "NarrowBand":
            sampels = (A @ signal) + noise 
            return sampels, signal, A, noise

        elif self.scenario == "Broadband":
            self.time_axis = System_model.time_axis
            samples = []
            SV = []
            self.f_sampling = System_model.f_sampling

            freq_axis = np.sort(-1 * np.linspace(-self.f_sampling // 2, -self.f_sampling // 2 , -self.f_sampling, endpoint=False))
            for idx,f in enumerate(freq_axis):
                A = np.array([self.SV_Creation(theta,f) for theta in self.DOA]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            sampels_time_domain = np.fft.ifft(samples.T, axis=1)[:, :self.T]
            return sampels_time_domain, signal, SV, noise

    def Noise_Creation(self, N_mean, N_Var):
        if self.scenario == "NarrowBand":
            # for NarrowBand scenario Noise represented in the time domain
            return np.sqrt(N_Var) * (np.random.randn(self.N, self.T) + 1j * np.random.randn(self.N, self.T)) + N_mean
        elif self.scenario == "Broadband":
            # for Broadband scenario Noise represented in the frequency domain
            noise = (np.sqrt(2) / 2) * (np.random.randn(self.N, len(self.time_axis)) + 1j * np.random.randn(self.N, len(self.time_axis))) + N_mean
            return np.fft.fft(noise)
    
    def Signal_creation(self, mode, S_mean = 0, S_Var = 1, SNR = 10, Carriers= None):
        if self.scenario == "NarrowBand":
            f = 1
            # Amp = 20 * np.log10(SNR)
            Amp = (10 ** (SNR / 10))

            if mode == "non-coherent": 
            # create M non - coherent signals
                return Amp * np.sqrt(S_Var) * (np.random.randn(self.M, self.T) + 1j * np.random.randn(self.M, self.T)) + S_mean
        
            elif mode == "coherent": 
                # create coherent signal such that all signals are the same
                # and arrived from different angels
                sig = Amp * np.sqrt(S_Var) * (np.random.randn(1, self.T) + 1j * np.random.randn(1, self.T)) + S_mean
                return np.repeat(sig, self.M, axis = 0)
        
        if self.scenario == "Broadband":
            Amp = (np.sqrt(2) / 2 ) * 20 * np.log10(SNR) 
            # Amp = (10 ** (SNR / 10))
            sig = []
            if mode == "non-coherent":
                for f_c in Carriers:
                    Amp_f_c = Amp * (np.random.randn() + 1j * np.random.randn())
                    sig_fc = Amp_f_c * np.exp(2 * np.pi * 1j * f_c * self.time_axis)
                    sig.append(sig_fc)
                # for Broadband scenario Noise represented in the frequency domain
                return np.fft.fft(sig)
            ###### should be updated ######
            if mode == "OFDM": 
                pass
            if mode == "mod-OFDM": 
                pass
        else:
            return 0


if __name__ == "__main__":
    print(create_closely_spaced_DOA(M=2, gap = 6))