from System_Model import *
import numpy as np
import scipy as sc
import scipy.signal
from useful_func import *
import matplotlib.pyplot as plt

class Model_Based_methods(object):
    def __init__(self,System_model):
        self.angels = np.linspace(-1 * np.pi / 2, np.pi / 2, 360, endpoint=False)                        # angle axis for represantation of the MUSIC spectrum
        self.system_model = System_model
        self.dist = System_model.dist
        self.M = System_model.M
        self.dist = System_model.dist

    def Classic_MUSIC(self, X, NUM_OF_SOURCES):
        '''
        Implamentation of the model-based MUSIC algorithm
        in Narrowband scenario.

        Input:
        @ X = sampels vector shape : Nx1
        @ NUM_OF_SOURCES = indication flag for the knowladge of number of sources
        
        Output:
        @ 

        '''
        if NUM_OF_SOURCES:                                                      # NUM_OF_SOURCES = TRUE : number of sources is given 
            M = self.M                                                
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues
            # clustring technique                                   
            pass
        R_x = np.cov(X)
        # print(R_x)                                                            # Create covariance matrix from sampels
        # R_x = autocor_mat(X, lag =0)                                          # Create covariance matrix from sampels
        eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
        Un = eigenvectors[:, M:]                                                # Take only the eigenvectors associated with Noise subspace 
        Spectrum,_= self.spectrum_calculation(Un)
        DOA_pred, _ = scipy.signal.find_peaks(Spectrum)                         # Find maximal values in spectrum
        DOA_pred = list(DOA_pred)
        DOA_pred.sort(key = lambda x: Spectrum[x], reverse = True)
        return DOA_pred, Spectrum, M                                                    # return estimated DOA

    def Classic_Root_MUSIC(self, X, NUM_OF_SOURCES, f=1):
        if NUM_OF_SOURCES:                                                              # NUM_OF_SOURCES = TRUE : number of sources is given
            M = self.M
        else:                                                                   # NUM_OF_SOURCES = False : M is given using  multiplicity of eigenvalues        
            # clustring technique
            pass
        R_x = np.cov(X)                                                         # Create covariance matrix from sampels
        eigenvalues, eigenvectors = np.linalg.eig(R_x)                          # Find the eigenvalues and eigenvectors using EVD
        Us, Un = eigenvectors[:, :M] , eigenvectors[:, M:]
        F = Un @ np.conj(Un).T                                                  # Set F as the matrix contains Information
        coeff = sum_of_diag(F)                                                  # Calculate the sum of the diagonals of F
        # print(coeff)
        roots = list(find_roots(coeff))                                         
                                                                                # By setting the diagonals as the coefficients of the polynomial
                                                                                # Calculate its roots
        # print(roots)
        roots.sort(key = lambda x : abs(abs(x) - 1))                            # Take only roots which are outside unit circle 
        roots1 = [root for root in roots if ((abs(root) - 1) < 0)][:M]
        # roots1 = roots[:2 * M:2]                                              # Take the M most closest to the unit circle roots
        roots_angels = np.angle(roots1)                                         # Calculate the phase component of the roots 
        DOA_pred = np.arcsin((1/(2 * np.pi * self.dist * f)) * roots_angels)    # Calculate the DOA our of the phase component
        DOA_pred = (180 / np.pi) * DOA_pred                                     # Convert from radians to Deegres
        roots_angels_all = np.angle(roots)                                      # Calculate the phase component of the roots 
        DOA_pred_all = np.arcsin((1/(2 * np.pi * self.dist * f)) * roots_angels_all)                              # Calculate the DOA our of the phase component
        DOA_pred_all = (180 / np.pi) * DOA_pred_all                                     # Convert from radians to Deegres
        return DOA_pred, roots, M, DOA_pred_all, roots_angels_all

    def spectrum_calculation(self, Un, f=1, Array_form="ULA"):
        Spectrum_equation = []
        for angle in self.angels:
            a = self.system_model.SV_Creation(theta= angle, f= f, Array_form = Array_form)
            Spectrum_equation.append(np.conj(a).T @ Un @ np.conj(Un).T @ a)
        Spectrum_equation = np.array(Spectrum_equation, dtype=np.complex)
        Spectrum = 1 / Spectrum_equation
        return Spectrum, Spectrum_equation




