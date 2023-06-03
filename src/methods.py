import numpy as np
import scipy.signal
import scipy
import torch
from src.system_model import *
from src.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelBasedMethods(object):
    def __init__(self, system_model: SystemModel):
        """Class for initializing the model based doa estimation methods  

        Args:
            system_model (): 
        """        
        self.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 360, endpoint=False)                        # angle axis for representation of the MUSIC spectrum
        self.Sys_Model = system_model
        self.M = system_model.M
        self.N = system_model.N
        self.dist = system_model.dist

    def broadband_MUSIC(self, X, NUM_OF_SOURCES=True, number_of_bins=50):
        number_of_bins = int(self.Sys_Model.max_freq["Broadband"] / 10)
        if NUM_OF_SOURCES:                                                      # NUM_OF_SOURCES = TRUE : number of sources is given 
            M = self.M                                                
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues
            # clustring technique                                   
            pass
        X = np.fft.fft(X, axis=1, n=self.Sys_Model.f_sampling["Broadband"])
        num_of_samples = len(self.angels)
        spectrum = np.zeros((number_of_bins, num_of_samples))
        
        for i in range(number_of_bins):
            ind = int(self.Sys_Model.min_freq["Broadband"]) + i * len(self.Sys_Model.f_rng["Broadband"]) // number_of_bins
            R_x = np.cov(X[:, ind:ind + len(self.Sys_Model.f_rng["Broadband"]) // number_of_bins])
            eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
            # Un = eigenvectors[:, M:]
            Un = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, M:]  
            spectrum[i], _= self.spectrum_calculation(Un, f=ind + len(self.Sys_Model.f_rng["Broadband"]) // number_of_bins - 1)
        
        # average spectra to one spectrum
        spectrum = np.sum(spectrum, axis=0)
        DOA_pred, _ = scipy.signal.find_peaks(spectrum)
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key = lambda x: spectrum[x], reverse = True)
        return DOA_pred, spectrum, M     

    def broadband_root_music(self, X, NUM_OF_SOURCES=True, number_of_bins=50):
        number_of_bins = int(self.Sys_Model.max_freq["Broadband"] / 10)
        if NUM_OF_SOURCES:                                                      # NUM_OF_SOURCES = TRUE : number of sources is given 
            M = self.M                                                
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues
            # clustring technique                                   
            pass
        X = np.fft.fft(X, axis=1, n=self.Sys_Model.f_sampling["Broadband"])
        num_of_samples = len(self.angels)
        F = []
        
        for i in range(number_of_bins):
            ind = int(self.Sys_Model.min_freq["Broadband"]) + i * len(self.Sys_Model.f_rng["Broadband"]) // number_of_bins
            R_x = np.cov(X[:, ind:ind + len(self.Sys_Model.f_rng["Broadband"]) // number_of_bins])
            eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
            # Un = eigenvectors[:, M:]
            Un = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, M:]  
            F.append(Un @ np.conj(Un).T) 
        
        # average spectra to one spectrum
        F_unified = np.sum(np.array(F), axis=0)/number_of_bins
        coeff = sum_of_diag(F_unified)                                                  # Calculate the sum of the diagonals of F
        roots = list(find_roots(coeff))                                         
                                                                                # By setting the diagonals as the coefficients of the polynomial
                                                                                # Calculate its roots
        roots.sort(key = lambda x : abs(abs(x) - 1))                             
        roots_inside = [root for root in roots if ((abs(root) - 1) < 0)][:M]    # Take only roots which are inside unit circle
        
        roots_angels = np.angle(roots_inside)                                   # Calculate the phase component of the roots 
        DOA_pred = np.arcsin((1/(2 * np.pi * 0.5)) * roots_angels)        # Calculate the DOA out of the phase component
        DOA_pred = (180 / np.pi) * DOA_pred                                     # Convert from radians to degrees
        
        roots_angels_all = np.angle(roots)                                      # Calculate the phase component of the roots 
        DOA_pred_all = np.arcsin((1/(2 * np.pi * 0.5)) * roots_angels_all)                              # Calculate the DOA our of the phase component
        DOA_pred_all = (180 / np.pi) * DOA_pred_all                                     # Convert from radians to Deegres
        return DOA_pred, roots, M, DOA_pred_all, roots_angels_all
        
    def MUSIC(self, X, NUM_OF_SOURCES=True, SPS=False, sub_array_size=0, DR=False, scenario='NarrowBand'):
        '''
        Implementation of the model-based MUSIC algorithm in Narrow-band scenario.
        
        Input:
        ------
        X = samples vector shape : Nx1
        NUM_OF_SOURCES = known number of sources flag
        SPS = pre-processing Spatial smoothing algorithm flag
        sub_array_size = size of sub array for spatial smoothing
        
        Returns:
        --------
        DOA_pred: the predicted DOA's
        spectrum: the MUSIC spectrum
        M: number of estimated/given sources
        
        '''
        if NUM_OF_SOURCES:                                                      # NUM_OF_SOURCES = TRUE : number of sources is given 
            M = self.M                                                
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues
            # clustring technique                                   
            pass
        
        if SPS:
            number_of_sensors = self.N
            number_of_sub_arrays = number_of_sensors - sub_array_size + 1
            
            ## Averaged covariance matrix
            R_x = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros((sub_array_size, sub_array_size))
            for j in range(number_of_sub_arrays):
                X_sub = X[j:j + sub_array_size,:]
                R_x += np.cov(X_sub)
            R_x /= number_of_sub_arrays
            # R_x = np.mean(np.array(Rx_sub_arrays), 0)
                
        else:
            R_x = np.cov(X)                                                     # Create covariance matrix from samples

        eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
        Un = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, M:]                                                # Take only the eigenvectors associated with Noise subspace 
        # Un = eigenvectors[:, M:]                                                # Take only the eigenvectors associated with Noise subspace 
        
        # Generate the MUSIC spectrum
        if scenario.startswith("Broadband"):
            # hard-coded for OFDM scenario within range [0-500] Hz
            f = 500 
        else:
            f = 1
        spectrum, _ = self.spectrum_calculation(Un, f = f)
        DOA_pred, _ = scipy.signal.find_peaks(spectrum)                         # Find maximal values in spectrum
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key = lambda x: spectrum[x], reverse = True)
        return DOA_pred, spectrum, M                                                    # return estimated DOA

    def root_music(self, X, NUM_OF_SOURCES=True, SPS=False, sub_array_size=0):
        '''
        Implementation of the model-based Root-MUSIC algorithm in Narrow-band scenario.
        
        Input:
        ------
        X = samples vector shape : Nx1
        NUM_OF_SOURCES = known number of sources flag
        SPS = pre-processing Spatial smoothing algorithm flag
        sub_array_size = size of sub array for spatial smoothing
        
        Returns:
        --------
        DOA_pred = the predicted DoA's
        roots = the roots of true DoA's 
        M = number of estimated/given sources
        roots_angels_all = all the roots produced by the algorithm
        DOA_pred_all = all the angels produced by the algorithm 
        '''
        if NUM_OF_SOURCES:                                                              # NUM_OF_SOURCES = TRUE : number of sources is given
            M = self.M
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues        
            # clustering technique
            threshold = 0.2
            norm_eigenvalues = eigenvalues / np.max(eigenvalues)
            # simplest clustering method: with threshold
            M = self.N - norm_eigenvalues[np.where(norm_eigenvalues < threshold)].shape(0)
        
        if SPS:
            number_of_sensors = self.N
            number_of_sub_arrays = number_of_sensors - sub_array_size + 1
            
            ## Averaged covariance matrix
            R_x = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros((sub_array_size, sub_array_size))
            for j in range(number_of_sub_arrays):
                X_sub = X[j:j + sub_array_size,:]
                R_x += np.cov(X_sub)
            R_x /= number_of_sub_arrays
            # R_x = np.mean(np.array(Rx_sub_arrays), 0)
        else:
            R_x = np.cov(X)                                                     # Create covariance matrix from samples
        
        eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
        Un = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, M:]
        # Un = eigenvectors[:, M:]

        F = Un @ np.conj(Un).T                                                  # Set F as the matrix contains Information
        coeff = sum_of_diag(F)                                                  # Calculate the sum of the diagonals of F
        roots = list(find_roots(coeff))                                         
                                                                                # By setting the diagonals as the coefficients of the polynomial
                                                                                # Calculate its roots
        roots.sort(key = lambda x : abs(abs(x) - 1))                             
        roots_inside = [root for root in roots if ((abs(root) - 1) < 0)][:M]    # Take only roots which are inside unit circle
        
        roots_angels = np.angle(roots_inside)                                   # Calculate the phase component of the roots 
        DOA_pred = np.arcsin((1/(np.pi)) * roots_angels)        # Calculate the DOA out of the phase component
        DOA_pred = (180 / np.pi) * DOA_pred                                     # Convert from radians to degrees
        
        roots_angels_all = np.angle(roots)                                      # Calculate the phase component of the roots 
        DOA_pred_all = np.arcsin((1/(np.pi)) * roots_angels_all)                              # Calculate the DOA our of the phase component
        DOA_pred_all = (180 / np.pi) * DOA_pred_all                                     # Convert from radians to Deegres
        return DOA_pred, roots, M, DOA_pred_all, roots_angels_all

    def spectrum_calculation(self, Un, f=1, array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.Sys_Model.steering_vec(theta= angle, f= f, array_form = array_form)
            a = a[:Un.shape[0]]                                         # sub-array response for Spatial smoothing 
            Spectrum_equation.append(np.conj(a).T @ Un @ np.conj(Un).T @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=complex)
        spectrum = 1 / Spectrum_equation
        return spectrum, Spectrum_equation
    
    def clustering(self, eigenvalues):
        """_summary_

        Args:
            eigenvalues (_type_): _description_

        Returns:
            _type_: _description_
        """        
        threshold = 0.2
        norm_eigenvalues = eigenvalues / np.max(eigenvalues)
        # simplest clustering method: with threshold
        return norm_eigenvalues[np.where(norm_eigenvalues < threshold)]
    
    #TODO: Add model-based broadband methods
    
    def hybrid_MUSIC(self, model_MUSIC, Rz, scenario=None):
        '''
    Implementation of the hybrid MUSIC algorithm.
    
    Input:
    @ Rz = samples vector shape : Nx1
    @ NUM_OF_SOURCES = known number of sources flag
    @ SPS = pre-processing Spatial smoothing algorithm flag
    @ sub_array_size = size of sub array for spatial smoothing
    
    Output:
    @ DOA_pred = the predicted DOA's
    @ spectrum = the MUSIC spectrum
    @ M = number of estimated/given sources

        '''
        # Generate model
        M = self.M
        
        # Predict the covariance matrix using the DR model 
        model_MUSIC.eval()
        R_x = model_MUSIC(Rz, M)[3]
        R_x = np.array(R_x.squeeze())
        
        # Find eigenvalues and eigenvectors given the hybrid covariance
        eigenvalues, eigenvectors = np.linalg.eig(R_x)

        # Define the Noise subspace
        Un = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, M:] 

        # Generate the MUSIC spectrum
        if scenario.startswith("Broadband"):
            # hard-coded for OFDM scenario within range [0-500] Hz
            f = 500 
        else:
            f = 1
        spectrum, _ = self.spectrum_calculation(Un, f = f)

        # Find peaks in the spectrum
        DOA_pred, _ = scipy.signal.find_peaks(spectrum)

        # Associate highest peaks with the DOA predictions
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key = lambda x: spectrum[x], reverse = True)

        # return estimated DOA, spectrum and number of sources
        return DOA_pred, spectrum, M
    
    def esprit(self, X, NUM_OF_SOURCES:bool=True, SPS:bool=False, sub_array_size=0,
               HYBRID = False, model_ESPRIT=None, Rz=None, scenario='NarrowBand'):
        '''
        Implementation of the model-based ESPRIT algorithm in Narrow-band scenario.
        
        Input:
        @ X = samples vector shape : Nx1
        @ NUM_OF_SOURCES = known number of sources flag
        @ SPS = pre-processing Spatial smoothing algorithm flag
        @ sub_array_size = size of sub array for spatial smoothing
        
        Output:
        @ DOA_pred = the predicted DoA's
        @ roots = the roots of true DoA's 
        @ M = number of estimated/given sources
        @ roots_angels_all = all the roots produced by the algorithm
        @ DOA_pred_all = all the angels produced by the algorithm 
        '''
        N = self.N
        if NUM_OF_SOURCES:                                                              # NUM_OF_SOURCES = TRUE : number of sources is given
            M = self.M
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues        
            # clustering technique
            pass
        
        if SPS:
            number_of_sensors = self.N
            number_of_sub_arrays = number_of_sensors - sub_array_size + 1
            
            ## Averaged covariance matrix
            R_x = np.zeros((sub_array_size, sub_array_size)) + 1j * np.zeros((sub_array_size, sub_array_size))
            for j in range(number_of_sub_arrays):
                X_sub = X[j:j + sub_array_size,:]
                R_x += np.cov(X_sub)
            R_x /= number_of_sub_arrays
            # R_x = np.mean(np.array(Rx_sub_arrays), 0)
        elif HYBRID:
            # Predict the covariance matrix using the DR model
            model_ESPRIT.eval()
            R_x = model_ESPRIT(Rz, M)[3]
            R_x = np.array(R_x.squeeze())
        else:
            R_x = np.cov(X)                                                     # Create covariance matrix from samples
        
        if scenario.startswith("Broadband"):
            # hard-coded for OFDM scenario within range [0-500] Hz
            f = 500
        else:
            f = 1
        
        eigenvalues, eigenvectors = np.linalg.eig(R_x)  # Apply eigenvalue decomposition (EVD)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]   # Sort eigenvectors based on eigenvalues order 
        Us, Un= eigenvectors[:, 0:M], eigenvectors[:, M:]   # Create distinction between noise and signal subspaces 
        # Us_upper, Us_lower = Us[0:N-1], Us[1:N]     # separate the first M columns of the signal subspace
        Us_upper, Us_lower = Us[0:R_x.shape[0]-1], Us[1:R_x.shape[0]]     # separate the first M columns of the signal subspace
        
        phi = np.linalg.pinv(Us_upper) @ Us_lower
        phi_eigenvalues, _ = np.linalg.eig(phi)                          # Find the eigenvalues and eigenvectors using EVD
        eigenvalues_angle = np.angle(phi_eigenvalues)                                   # Calculate the phase component of the roots 
        DOA_pred = -np.arcsin((1/(2 * np.pi * self.dist[scenario] * f)) * eigenvalues_angle)        # Calculate the DOA out of the phase component
        DOA_pred = (180 / np.pi) * DOA_pred                                     # Convert from radians to degrees
        return DOA_pred, M

    
    def MVDR(self, X, NUM_OF_SOURCES:bool=True, SPS:bool=False, sub_array_size=0,
               HYBRID = False, model_mvdr=None, Rz=None, scenario='NarrowBand', eps = 0):
        '''
        Implementation of the Minimum Variance beamformer algorithm
        in narrow-band scenario, while applying spatial-smoothing

        Input
        -------------
        X: samples vector shape : Nx1
        
        Output:
        -------------
        response_curve: the response curve of the MVDR beamformer  
        
        '''
        if NUM_OF_SOURCES:                                                              # NUM_OF_SOURCES = TRUE : number of sources is given
            M = self.M
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues        
            # clustering technique
            pass
        response_curve = []
        for angle in self.angels:
            if HYBRID:
                # Predict the covariance matrix using the DR model
                model_mvdr.eval()
                R_x = model_mvdr(Rz, M)[3]
                R_x = np.array(R_x.squeeze())
            else:
                R_x = np.cov(X)                                                     # Create covariance matrix from samples

            ## Diagonal Loading
            R_eps_MVDR = R_x + eps * np.trace(R_x) * np.identity(R_x.shape[0])
        
            ## BeamForming response calculation 
            R_inv = np.linalg.inv(R_eps_MVDR)
            if scenario.startswith("Broadband"):
            # hard-coded for OFDM scenario within range [0-500] Hz
                f = 500 
            else:
                f = 1
            a = self.Sys_Model.steering_vec(theta = angle, f= f, array_form = "ULA").reshape((self.N,1))
            
            ## Adaptive calculation of W_opt
            W_opt = (R_inv @ a) / (np.conj(a).T @ R_inv @ a)
            # W_opt = 1 / (np.conj(a).T @ R_inv @ a)
            
            response_curve.append(((np.conj(W_opt).T @ R_eps_MVDR @ W_opt).reshape((1))).item())
            # response_curve.append(W_opt.item())
            
        response_curve = np.array(response_curve, dtype=complex)
        return response_curve

def root_music_torch(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix. 

    Args:
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """        
    
    dist = 0.5 
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Noise subspace as the eigenvectors which associated with the M greatest eigenvalues 
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:] 
        # Generate hermitian noise subspace matrix 
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)                                                            
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        
        ## Calculates angles of all roots
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots) 
        # Calculate doa
        doa_pred_all = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        
        # Take only roots which outside the unit circle
        roots = roots[sorted(range(roots.shape[0]), key = lambda k : abs(abs(roots[k]) - 1))]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)
        
    return torch.stack(doa_batches, dim = 0), torch.stack(doa_all_batches, dim = 0), roots_to_return