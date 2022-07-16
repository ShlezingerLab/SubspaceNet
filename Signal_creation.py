import numpy as np
from System_Model import *

def CreateDOAwithGapM(M, gap):
    while(True):
        # DOA = np.round(np.random.rand(M) *  180 ,decimals = 0) - 90.00
        DOA = np.round(np.random.rand(M) *  180 ,decimals = 2) - 90.00
        # DOA = np.random.randint(180, size = M) - 90
        DOA.sort()
        diffbetweenAngles = np.array([np.abs(DOA[i+1] - DOA[i]) for i in range(M-1)])
        if(np.sum(diffbetweenAngles > gap) == M - 1 and np.sum(diffbetweenAngles < (180 - gap)) == M - 1):
            break
    return DOA

class Sampels(object):
    def __init__(self, System_model, DOA, observations):
        self.scenario = System_model.scenario
        self.N = System_model.N
        self.M = System_model.M
        self.T = observations
        self.SV_Creation = System_model.SV_Creation
        if DOA == None:
          # self.DOA = np.array(np.pi * (np.random.rand(self.M) - 0.5))         # generate aribitrary DOA angels
          self.DOA = (np.pi / 180) * np.array(CreateDOAwithGapM(M = self.M, gap = 15)) # (~0.2 rad)
            # self.DOA = np.array(np.round((np.pi * ((np.random.rand(self.M) - 0.5))),decimals=2))
        else: 
          self.DOA = (np.pi / 180) * np.array(DOA)                              # define DOA angels
    
    def Sampels_creation(self, mode, N_mean= 0, N_Var= 1, S_mean= 0, S_Var= 1, SNR= 10, Carriers= None):
        '''
        @mode = represent the specific mode in the specific scnarion
                e.g. "Broadband" scenarion in "non-coherent" mode
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


#********************#
#   create dataset   #
#********************#
def create_dataset(name, size, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, N, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, M))
    for i in tqdm(range(size)):
        thetas = np.pi * (np.random.rand(d) - 1/2)  # random source directions
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


#**************************#
#   create mixed dataset   #
#**************************#
def create_mixed_dataset(name, first, second, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param first -- The path/name of the first dataset to be mixed with...
        @param second -- The path/name of the second dataset.
        @param save -- If true the dataset is saved to filename.
    """
    hf1 = h5py.File(first + '.h5', 'r')
    hf2 = h5py.File(second + '.h5', 'r')

    dataX1 = np.array(hf1.get('X'))
    dataY1 = np.array(hf1.get('Y'))

    dataX2 = np.array(hf2.get('X'))
    dataY2 = np.array(hf2.get('Y'))

    dataX = np.concatenate((dataX1, dataX2), axis=1)
    dataY = np.concatenate((dataY1, dataY2), axis=1)

    dataX, dataY = utils.shuffle(dataX, dataY)

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=dataX)
        hf.create_dataset('Y', data=dataY)
        hf.close()

    return dataX, dataY
