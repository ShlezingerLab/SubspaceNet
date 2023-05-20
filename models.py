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
        self.LeakyReLU = nn.LeakyReLU(ActivationVal)
        # self.LeakyReLU = nn.Tanh()
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
            Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:] 
            F = torch.matmul(Un, torch.t(torch.conj(Un)))                                           # Set F as the matrix conatains Information, 
            coeff = self.sum_of_diags(F)                                                            # Calculate the sum of the diagonals of F
            roots = self.find_roots(coeff)                                                          # Calculate its roots
            
            roots_angels_all = torch.angle(roots)                                                   # Calculate the phase component of the roots 
            DOA_pred_all = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels_all)              # Calculate the DOA our of the phase component
            DOA_all_list.append(DOA_pred_all)
            
            roots = roots[sorted(range(roots.shape[0]), key = lambda k : abs(abs(roots[k]) - 1))]   # Take only roots which are outside unit circle
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred_test = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                 # Calculate the DOA our of the phase component
            roots_to_return = roots

            mask = (torch.abs(roots) - 1) < 0
            roots = roots[mask][:M]

            
            
            
            # roots = roots[:2 * M:2]                                                                 # Take the M most closest to the unit circle roots
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                      # Calculate the DOA our of the phase component
            # print("DOA Pred", DOA_pred * 180 / np.pi)
            DOA_list.append(DOA_pred)                                                               # Convert from radians to Deegres
        return torch.stack(DOA_list, dim = 0), torch.stack(DOA_all_list, dim = 0), roots_to_return, DOA_pred_test
    
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
        DOA, DOA_all, roots, sorted_angels = self.Root_MUSIC(Rz, M)                                                  # Output shape [Batch size, M]
        return DOA, DOA_all, roots, Rz, sorted_angels

class Deep_Root_Net_AntiRectifier(nn.Module):
    def __init__(self, tau):
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
        # self.LeakyReLU = nn.LeakyReLU(ActivationVal)
        # self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(0.2)

    def AntiRectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)
    
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
            Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:] 
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
            mask = (torch.abs(roots) - 1) < 0
            
            roots = roots[mask][:M]
            roots_angels = torch.angle(roots)                                                       # Calculate the phase component of the roots 
            DOA_pred = torch.arcsin((1/(2 * np.pi * dist * f)) * roots_angels)                      # Calculate the DOA our of the phase component
            DOA_list.append(DOA_pred)                                                               # Convert from radians to Deegres
        
            eigenvalues = torch.real(eigenvalues) / torch.max(torch.real(eigenvalues))
            # eigenvalues = torch.real(eigenvalues)
            norm_eig = torch.flip(torch.sort(eigenvalues)[0], (0,))
            # eig_diffs.append((norm_eig[0] - norm_eig)[1])
            minimal_signal_eig = norm_eig[M-1] - norm_eig[-1]
            maximal_noise_eig = norm_eig[M] - norm_eig[-1]
            # print(eigenvalues)
            # print(norm_eig[M-1] - norm_eig[-1], norm_eig[M] - norm_eig[-1])
            
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
        
        ## AutoEncoder Architecture
        x = self.conv1(New_Rx_tau)
        x = self.AntiRectifier(x)

        x = self.conv2(x)
        x = self.AntiRectifier(x)
        
        x = self.conv3(x)
        x = self.AntiRectifier(x)

        x = self.deconv2(x)
        x = self.AntiRectifier(x)

        x = self.deconv3(x)
        x = self.AntiRectifier(x)

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
        DOA, DOA_all, roots = self.Root_MUSIC(Rz, M)                      # Output shape [Batch size, M]
        return DOA, DOA_all, roots, Rz
    
class Deep_Augmented_MUSIC(nn.Module):
    def __init__(self, N, T, M):
        ## input dim (N, T)
        super(Deep_Augmented_MUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi / 2, np.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        # nn.init.xavier_uniform(self.rnn.weigh)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        nn.init.xavier_uniform(self.fc.weight)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        nn.init.xavier_uniform(self.fc3.weight)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
    
    def steering_vec(self):
        sv = []
        for angle in self.angels:
            a = torch.exp(-1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle))
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un, f=1, array_form="ULA"):
        Spectrum_equation = []
        for i in range(self.angels.shape[0]):
            Spectrum_equation.append(torch.real(torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i]))
            # Spectrum_equation.append(torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i])
        Spectrum_equation = torch.stack(Spectrum_equation, dim=0)
        # spectrum = 1 / torch.abs(Spectrum_equation)
        spectrum = 1 / Spectrum_equation
        
        return spectrum, Spectrum_equation

    def pre_MUSIC(self, Rz):
        spectrum = []
        Bs_Rz = Rz
        for iter in range(self.BATCH_SIZE):
            R = Bs_Rz[iter]
            eigenvalues, eigenvectors = torch.linalg.eig(R)                                         # Find the eigenvalues and eigenvectors using EVD
            # Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, self.M:]
            Un = eigenvectors[:, self.M:]
            spectrum.append(self.spectrum_calculation(Un)[0])             
        # TODO: error here
        return torch.stack(spectrum, dim = 0)
    
    def forward(self, X):
        # input X.shape == [Batch size, N, T]
        X = torch.cat((torch.real(X),torch.imag(X)), 1) # X.shape ==  [Batch size, 2N, T]
        self.BATCH_SIZE = X.shape[0]
        X = X.view(X.size(0), X.size(2), X.size(1)) # [Batch size, T, 2N]
        X = self.BatchNorm(X)
        gru_out, hn = self.rnn(X)
        # Rx = gru_out[:, -1, :]
        Rx = gru_out[:,-1]
        # Rx = hn.squeeze()
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1)) # [Batch size, T, 2N]
        Rx = self.fc(Rx)

        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, :self.N, :]                                                   # Output shape [Batch size, N, N])  
        Rx_imag = Rx_view[:, self.N:, :]                                                   # Output shape [Batch size, N, N])  
        Kx_tag = torch.complex(Rx_real, Rx_imag)                                           # Output shape [Batch size, N, N])

        spectrum = self.pre_MUSIC(Kx_tag)
        
        y = self.ReLU(self.fc1(spectrum))
        y = self.ReLU(self.fc2(y))
        y = self.ReLU(self.fc2(y))
        
        DOA = self.fc3(y)
        return DOA
    
class CNN_DOA(nn.Module):
    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(CNN_DOA, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size = 3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size = 2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
    
    def forward(self, X):
        # input X.shape == [Batch size, 3, N, N]
        X = X.view(X.size(0), X.size(3), X.size(2), X.size(1)) # [Batch size, 3, N, N]
        # conv layer 1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        # X = self.BatchNorm(X)
        X = self.ReLU(X)
        
        # conv layer 2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        # X = self.BatchNorm(X)
        X = self.ReLU(X)
        
        # conv layer 3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        # X = self.BatchNorm(X)
        X = self.ReLU(X)
        
        # conv layer 4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        # X = self.BatchNorm(X)
        X = self.ReLU(X)
        
        # FC layer
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))
        X = self.DropOut(self.ReLU(self.fc2(X)))
        X = self.DropOut(self.ReLU(self.fc3(X)))
        X = self.fc4(X)
        X = self.Sigmoid(X)
        
        return X


if __name__ == "__main__":
    # model = Deep_Augmented_MUSIC(N=8, T=10, M=2)
    # X = torch.rand((8, 10)) + 1j * torch.rand((8, 10))
    # X_new = X
    # for i in range(10000):
    #     X_new = torch.stack([X_new, X], dim =0)
    # print(X_new.shape)
    # y = model(X_new)
    # print(y)
    
    model = CNN_DOA(N=8, grid_size=361)
    X = torch.rand((10, 8, 8, 3))
    # X_new = torch.tensor(X, dtype=torch.complex64)
    X_new = X
    Y = torch.zeros(181)
    Y[10] = 1
    Y[20] = 1
    y = model(X_new)