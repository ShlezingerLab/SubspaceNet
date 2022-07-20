import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.simplefilter("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class Deep_Root_Net(nn.Module):
    def __init__(self, tau, ActivationVal):
        self.tau = tau
        super(Deep_Root_Net, self).__init__()
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 2)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size= 2)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size= 2)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size= 2)
        # self.ReLU = nn.ReLU()
        # self.SeLU = nn.SELU()
        # self.LeakyReLU = nn.LeakyReLU(ActivationVal)
        self.LeakyReLU = nn.Tanh()
        # self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(0.2)

    def sum_of_diags(self, Matrix):
        coeff =[]
        diag_index = torch.linspace(-Matrix.shape[0] + 1, Matrix.shape[0] - 1, (2 * Matrix.shape[0]) - 1, dtype = int)
        for idx in diag_index:
            coeff.append(torch.sum(torch.diagonal(Matrix, idx)))
        return torch.stack(coeff, dim = 0)

    def find_roots(self, coeff):
        A_torch = torch.diag(torch.ones(len(coeff)-2,dtype=coeff.dtype), -1)
        A_torch[0,:] = -coeff[1:] / coeff[0]
        roots = torch.linalg.eigvals(A_torch)
        return roots

    def Root_MUSIC(self, Rz, M):
        dist = 0.5 
        f = 1
        DOA_list = []
        DOA_all_list = []
        Bs_Rz = Rz
        for iter in range(self.BATCH_SIZE):
            R = Bs_Rz[iter]
            eigenvalues, eigenvectors = torch.linalg.eig(R)                                         # Find the eigenvalues and eigenvectors using EVD
            Un = eigenvectors[:, M:]
            F = torch.matmul(Un, torch.t(torch.conj(Un)))                                           # Set F as the matrix conatains Information, 
            coeff = self.sum_of_diags(F)                                                            # Calculate the sum of the diagonals of F
            roots = self.find_roots(coeff)                                                          # Calculate its roots
            
            roots_angels_all = torch.angle(roots)                                                   # Calculate the phase component of the roots 
            DOA_pred_all = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels_all)              # Calculate the DOA our of the phase component
            DOA_all_list.append(DOA_pred_all)
            roots_to_return = roots
            
            roots = roots[sorted(range(roots.shape[0]), key = lambda k : abs(abs(roots[k]) - 1))]   # Take only roots which are outside unit circle
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred_test = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                 # Calculate the DOA our of the phase component
            
            # print("abs(roots)", abs(roots)-1)
            # print("DOA_pred_test", DOA_pred_test * 180 / np.pi)
            
            mask = (torch.abs(roots) - 1) < 0
            roots = roots[mask][:M]
            # indices = torch.nonzero(mask
            # print("abs(roots)", abs(roots)-1)

            
            
            # roots = [root for root in roots if (abs(root) - 1) < 0][:M]
            # roots = roots[:2 * M:2]                                                                 # Take the M most closest to the unit circle roots
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                      # Calculate the DOA our of the phase component
            # print("DOA Pred", DOA_pred * 180 / np.pi)
            DOA_list.append(DOA_pred)                                                               # Convert from radians to Deegres
        return torch.stack(DOA_list, dim = 0), torch.stack(DOA_all_list, dim = 0), roots_to_return
    
    def Gramian_matrix(self, Kx, eps):
        '''
        multiply a Matrix Kx with its Hermitian Conjecture,
        and adds eps to diagonal Value of the Matrix,
        In order to Ensure Hermit and PSD:
        Kx = (Kx)^H @ (Kx) + eps * I
        @ Kx(input) - Complex matrix with shape [BS, N, N]
        @ eps(input) - Multiplies constant added to each diangonal 
        @ Kx_Out - Hermit and PSD matrix with shape [BS, N, N]
        '''
        Kx_list = []
        Bs_kx = Kx
        for iter in range(self.BATCH_SIZE):
            K = Bs_kx[iter]
            Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)                                       # output size(NxN)
            eps_Unit_Mat = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
            Rz = Kx_garm + eps_Unit_Mat                                                             # output size(NxN)
            Kx_list.append(Rz)
        Kx_Out = torch.stack(Kx_list, dim = 0)
        return Kx_Out
    
    def forward(self, New_Rx_tau, M):
        ## Input shape of signal X(t): [Batch size, N, T]
        self.N = New_Rx_tau.shape[-1]
        self.BATCH_SIZE = New_Rx_tau.shape[0]

        # New_Rx_tau = self.Create_Autocorr_tensor(X, self.tau).to(torch.float)         # Output shape [Batch size, tau, 2N, N]
        
        ## AutoEncoder Archtecture
        x = self.conv1(New_Rx_tau)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.conv3(x)
        x = self.LeakyReLU(x)

        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        x = self.deconv3(x)
        x = self.LeakyReLU(x)
        x = self.DropOut(x)
        Rx = self.deconv4(x)  
        Rx_View = Rx.view(Rx.size(0),Rx.size(2),Rx.size(3))                           # Output shape [Batch size, 2N, N]

        ## Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :self.N, :]                                               # Output shape [Batch size, N, N])  
        Rx_imag = Rx_View[:, self.N:, :]                                              # Output shape [Batch size, N, N])  
        Kx_tag = torch.complex(Rx_real, Rx_imag)                                      # Output shape [Batch size, N, N])

        ## Apply Gramian transformation to ensure Hermitian and PSD marix
        Rz = self.Gramian_matrix(Kx_tag, eps= 1)                                           # Output shape [Batch size, N, N]

        ## Rest of Root MUSIC algorithm
        # print(Rz)
        DOA, DOA_all, roots = self.Root_MUSIC(Rz, M)                                                  # Output shape [Batch size, M]
        return DOA, DOA_all, roots



class Deep_Root_Net_AntiRectifier(nn.Module):
    def __init__(self, tau, ActivationVal):
        self.tau = tau
        super(Deep_Root_Net_AntiRectifier, self).__init__()
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size = 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 2)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size = 2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size= 2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size= 2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size= 2)
        self.ReLU = nn.ReLU()
        # self.SeLU = nn.SELU()
        self.LeakyReLU = nn.LeakyReLU(ActivationVal)
        # self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(0.2)

    def AntiRectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)
        # print(X.shape)
        # meu = torch.mean(X)
        # meu = torch.mean(X, 2).repeat(X.shape[2],1)
        # norm_ava_X = (X - meu.T) / torch.linalg.norm(X - meu.T)
        # return torch.cat((self.ReLU(norm_ava_X), self.ReLU(-norm_ava_X)),1)
    
    def sum_of_diags(self, Matrix):
        coeff =[]
        diag_index = torch.linspace(-Matrix.shape[0] + 1, Matrix.shape[0] - 1, (2 * Matrix.shape[0]) - 1, dtype = int)
        for idx in diag_index:
            coeff.append(torch.sum(torch.diagonal(Matrix, idx)))
        return torch.stack(coeff, dim = 0)

    def find_roots(self, coeff):
        A_torch = torch.diag(torch.ones(len(coeff)-2,dtype=coeff.dtype), -1)
        A_torch[0,:] = -coeff[1:] / coeff[0]
        roots = torch.linalg.eigvals(A_torch)
        return roots

    def Root_MUSIC(self, Rz, M):
        dist = 0.5 
        f = 1
        DOA_list = []
        DOA_all_list = []
        Bs_Rz = Rz
        for iter in range(self.BATCH_SIZE):
            R = Bs_Rz[iter]
            eigenvalues, eigenvectors = torch.linalg.eig(R)                                         # Find the eigenvalues and eigenvectors using EVD
            Un = eigenvectors[:, M:]
            # print(Un[0].shape)
            # print(torch.angle(Un[0]))
            # Un = Un * torch.exp(-1j * torch.angle(Un[0]))
            # print(torch.angle(Un[0]))
            F = torch.matmul(Un, torch.t(torch.conj(Un)))                                           # Set F as the matrix conatains Information, 
            coeff = self.sum_of_diags(F)                                                            # Calculate the sum of the diagonals of F
            roots = self.find_roots(coeff)                                                          # Calculate its roots
            
            roots_angels_all = torch.angle(roots)                                                   # Calculate the phase component of the roots 
            DOA_pred_all = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels_all)              # Calculate the DOA our of the phase component
            DOA_all_list.append(DOA_pred_all)
            roots_to_return = roots
            
            # print("roots before sorting", roots)
            roots = roots[sorted(range(roots.shape[0]), key = lambda k : abs(abs(roots[k]) - 1))]   # Take only roots which are outside unit circle
            # print("roots after sorting", roots)
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred_test = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                 # Calculate the DOA our of the phase component
            
            # print("abs(roots)", abs(roots)-1)
            # print("DOA_pred_test", DOA_pred_test * 180 / np.pi)
            
            mask = (torch.abs(roots) - 1) < 0
            roots = roots[mask][:M]
            # indices = torch.nonzero(mask
            # print("abs(roots)", abs(roots)-1)

            
            
            # roots = [root for root in roots if (abs(root) - 1) < 0][:M]
            # roots = roots[:2 * M:2]                                                                 # Take the M most closest to the unit circle roots
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                      # Calculate the DOA our of the phase component
            # print("DOA Pred", DOA_pred * 180 / np.pi)
            DOA_list.append(DOA_pred)                                                               # Convert from radians to Deegres
        return torch.stack(DOA_list, dim = 0), torch.stack(DOA_all_list, dim = 0), roots_to_return
    
    def Gramian_matrix(self, Kx, eps):
        '''
        multiply a Matrix Kx with its Hermitian Conjecture,
        and adds eps to diagonal Value of the Matrix,
        In order to Ensure Hermit and PSD:
        Kx = (Kx)^H @ (Kx) + eps * I
        @ Kx(input) - Complex matrix with shape [BS, N, N]
        @ eps(input) - Multiplies constant added to each diangonal 
        @ Kx_Out - Hermit and PSD matrix with shape [BS, N, N]
        '''
        Kx_list = []
        Bs_kx = Kx
        for iter in range(self.BATCH_SIZE):
            K = Bs_kx[iter]
            Kx_garm = torch.matmul(torch.t(torch.conj(K)), K).to(device)                                       # output size(NxN)
            eps_Unit_Mat = (eps * torch.diag(torch.ones(Kx_garm.shape[0]))).to(device)
            Rz = Kx_garm + eps_Unit_Mat                                                             # output size(NxN)
            Kx_list.append(Rz)
        Kx_Out = torch.stack(Kx_list, dim = 0)
        return Kx_Out
    
    def forward(self, New_Rx_tau, M):
        ## Input shape of signal X(t): [Batch size, N, T]
        self.N = New_Rx_tau.shape[-1]
        self.BATCH_SIZE = New_Rx_tau.shape[0]

        # New_Rx_tau = self.Create_Autocorr_tensor(X, self.tau).to(torch.float)         # Output shape [Batch size, tau, 2N, N]
        
        ## AutoEncoder Archtecture
        x = self.conv1(New_Rx_tau)
        x = self.AntiRectifier(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.AntiRectifier(x)
        # print(x.shape)

        x = self.AntiRectifier(self.conv3(x))
        # print(x.shape)

        x = self.deconv2(x)
        x = self.AntiRectifier(x)
        # print(x.shape)

        x = self.AntiRectifier(self.deconv3(x))
        # print(x.shape)

        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # print(Rx.shape)

        Rx_View = Rx.view(Rx.size(0),Rx.size(2),Rx.size(3))                           # Output shape [Batch size, 2N, N]

        ## Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :self.N, :]                                               # Output shape [Batch size, N, N])  
        Rx_imag = Rx_View[:, self.N:, :]                                              # Output shape [Batch size, N, N])  
        Kx_tag = torch.complex(Rx_real, Rx_imag)                                      # Output shape [Batch size, N, N])

        ## Apply Gramian transformation to ensure Hermitian and PSD marix
        Rz = self.Gramian_matrix(Kx_tag, eps= 1)                                           # Output shape [Batch size, N, N]

        ## Rest of Root MUSIC algorithm
        # print(Rz)
        DOA, DOA_all, roots = self.Root_MUSIC(Rz, M)                                                  # Output shape [Batch size, M]
        return DOA, DOA_all, roots

