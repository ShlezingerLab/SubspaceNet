"""Subspace-Net plotting script 
    Details
    ------------
    Name: plotting.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 01/04/23

    Purpose
    ------------
    This script generates the plots which presented in the following papers: SubspaceNet journal paper:
    [1] D. H. Shmuel, J. Merkofer, G. Revach, R. J. G. van Sloun, and N. Shlezinger “Deep Root MUSIC Algorithm for Data-Driven DoA Estimation”, IEEE ICASSP 2023 
    [2] "SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation"
    
    The script uses the following functions:
        * create_dataset: For creating training and testing datasets 
        * run_simulation: For training DR-MUSIC model
        * evaluate_model: For evaluating subspace hybrid models

    This script requires that plot_style.txt will be included in the working repository.

"""
###############
#   Imports   #
###############
import numpy as np
from matplotlib import pyplot as plt

######################
# Conversion methods #
######################

def unit(value):
    return value

def rad2deg(value):
    return value * 180 / np.pi 

def deg2rad(value):
    return value * np.pi / 180

def rad2dB(mse_value:float):
    """Converts MSE value in radian scale into dB scale

    Args:
        mse_value (float): value of error in rad

    Returns:
        float: value of error in dB
    """    
    return 10 * np.log10(mse_value ** 2)

def scenario_to_plot(simulation, mode = "non-coherent", T = 200):
    Loss = {}
    x_axis = {}
    
    # Criterion
    RMSPE = True
    
    ## Simulation flags
    # simulation = "SNR"
    # simulation = "coherent basic"
    # simulation = "distance_calibration"
    # simulation = "sv calibration"
    # simulation = "OFDM"
    
    # mode = "non-coherent"
    # mode = "non-coherent"
    
    ## Observations
    # T = 200
    
    ##### SNR simulation #####
    if simulation.startswith("SNR"):
        # T = 200
        if T == 200:
            x_axis["SNR"] = np.array([-5, -4, -3, -2, -1])
            # non-coherent sources
            if mode.startswith("non-coherent"):            
                # RMSPE Scores
                if RMSPE:
                    Loss["MUSIC"]            = np.array([0.097, 0.0773, 0.0626, 0.0499, 0.04])
                    Loss["R-MUSIC"]          = np.array([0.073, 0.055, 0.0409, 0.0304, 0.022])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0399, 0.0297, 0.0228, 0.0164, 0.0133])  
                # MSPE Scores
                else:
                    Loss["MUSIC"]            = np.array([0.0503, 0.0400, 0.0321, 0.0302, 0.0196])
                    Loss["R-MUSIC"]          = np.array([0.0312, 0.0214, 0.0148, 0.0132, 0.0062])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0040, 0.0032, 0.0014, 0.0007, 0.0005])
            
            # coherent sources
            elif mode.startswith("coherent"):
                # RMSPE Scores            
                if RMSPE:
                    Loss["MUSIC"]            = np.array([0.202, 0.1915, 0.185, 0.18, 0.176])
                    Loss["R-MUSIC"]          = np.array([0.24, 0.2329, 0.225, 0.221, 0.219])
                    Loss["SPS+MUSIC"]        = np.array([0.162, 0.124, 0.106, 0.093, 0.084])
                    Loss["SPS+R-MUSIC"]      = np.array([0.149, 0.113, 0.095, 0.08, 0.07])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.027, 0.0199, 0.017, 0.0146, 0.011])
                
                ## MSPE Scores
                else:
                    Loss["MUSIC"]            = np.array([0.1124, 0.1055, 0.0954, 0.0976, 0.0948])
                    Loss["R-MUSIC"]          = np.array([0.1441, 0.1369, 0.1334, 0.1279, 0.1258])
                    Loss["SPS+MUSIC"]        = np.array([0.0942, 0.0723, 0.0638, 0.0569, 0.0531])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0829, 0.0619, 0.0536, 0.0443, 0.0394])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0031, 0.0011, 0.0007, 0.0005, 0.0003])
        elif T == 20:
            x_axis["SNR"] = np.array([5, 6, 7, 8, 9 ,10])
            # non-coherent sources
            if mode.startswith("non-coherent"):            
                pass
            # coherent sources
            elif mode.startswith("coherent"):
                # RMSPE Scores            
                if RMSPE:
                    Loss["MUSIC"]            = np.array([0.174, 0.171, 0.170, 0.169, 0.169, 0.169])
                    Loss["R-MUSIC"]          = np.array([0.218, 0.216, 0.216, 0.216, 0.217, 0.217])
                    Loss["SPS+MUSIC"]        = np.array([0.067, 0.0598, 0.052, 0.047, 0.042, 0.04])
                    Loss["SPS+R-MUSIC"]      = np.array([0.049, 0.043, 0.037, 0.0323, 0.027, 0.024])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0088, 0.0075, 0.0075, 0.0078, 0.0066, 0.006])
                
                ## MSPE Scores
                else:
                    Loss["MUSIC"]            = np.array([0.0935, 0.0917, 0.0903, 0.0898, 0.0900, 0.0856])
                    Loss["R-MUSIC"]          = np.array([0.1262, 0.1245, 0.1243, 0.1245, 0.1253, 0.1228])
                    Loss["SPS+MUSIC"]        = np.array([0.0259, 0.0234, 0.0194, 0.0167, 0.0143, 0.0120])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0428, 0.0383, 0.0336, 0.0305, 0.0283, 0.0267])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0000])
        elif T == 2:
            x_axis["SNR"] = np.array([5, 6, 7, 8, 9 ,10])
            # non-coherent sources
            if mode.startswith("non-coherent"):            
                # RMSPE Scores
                if RMSPE:
                    Loss["MUSIC"]           = np.array([0.479, 0.471, 0.465, 0.468, 0.458, 0.4609])
                    Loss["R-MUSIC"]         = np.array([0.424, 0.423, 0.416, 0.423, 0.412, 0.412])
                    Loss["SPS+MUSIC"]       = np.array([0.1827, 0.161, 0.141, 0.124, 0.11, 0.098])
                    Loss["SPS+R-MUSIC"]     = np.array([0.156, 0.132, 0.115, 0.094, 0.08,0.07])
                    Loss["SubNet+R-MUSIC"]  = np.array([0.05, 0.0418, 0.035, 0.0298, 0.0271, 0.024])
                # MSPE Scores
                else:
                    Loss["MUSIC"]           = np.array([0.2697, 0.2694, 0.2635, 0.2708, 0.2584, 0.2610])
                    Loss["R-MUSIC"]         = np.array([0.3189, 0.3117, 0.3083, 0.3089, 0.3002, 0.3045])
                    Loss["SPS+MUSIC"]       = np.array([0.1101, 0.0981, 0.0861, 0.0765, 0.0687, 0.0614])
                    Loss["SPS+R-MUSIC"]     = np.array([0.0848, 0.0714, 0.0582, 0.0484, 0.0407, 0.0346])
                    Loss["SubNet+R-MUSIC"]  = np.array([0.0117, 0.0087, 0.0064, 0.0045, 0.0035, 0.0032])
            
            # coherent sources
            elif mode.startswith("coherent"):
                # RMSPE Scores            
                if RMSPE:
                    Loss["MUSIC"]            = np.array([0.383, 0.354, 0.348, 0.322, 0.325, 0.314])
                    Loss["R-MUSIC"]          = np.array([0.46, 0.447, 0.449, 0.432, 0.437, 0.43])
                    Loss["SPS+MUSIC"]        = np.array([0.1657, 0.141, 0.125, 0.11, 0.1, 0.088])
                    Loss["SPS+R-MUSIC"]      = np.array([0.1444, 0.1196, 0.1025, 0.087, 0.075,0.065])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.033, 0.027, 0.024, 0.02, 0.017, 0.015])
                
                ## MSPE Scores
                else:
                    Loss["MUSIC"]            = np.array([0.2471, 0.2231, 0.2196, 0.1980, 0.2023, 0.1894])
                    Loss["R-MUSIC"]          = np.array([0.3179, 0.3027, 0.3064, 0.2949, 0.2988, 0.2917])
                    Loss["SPS+MUSIC"]        = np.array([0.0972, 0.0815, 0.0734, 0.0653, 0.0603, 0.0534])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0767, 0.0609, 0.0513, 0.0422, 0.0362, 0.0316])
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0068, 0.0049, 0.0040, 0.0036, 0.0031, 0.0027])
    ###### BroadBand simulation  #####
    elif simulation.startswith("OFDM"):
        x_axis["Observations"] = [50, 100, 200, 1000]
        if RMSPE:
            Loss["BB-MUSIC"]        = np.array([0.34, 0.306 ,0.308  ,0.186])
            Loss["DA-MUSIC"]        = np.array([0.192, 0.125, 0.073 ,0.074])
            Loss["SubNet+R-MUSIC"]  = np.array([0.088 , 0.056  ,0.0517 ,0.039])
            # Add here music and hybrid music

    ###### mis-calibration simulations  ######
    elif simulation.startswith("distance_calibration"):
        x_axis["eta"] = [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075]
        if RMSPE:
            Loss["R-MUSIC"]         = np.array([0.0398, 0.0741, 0.1030, 0.1410, 0.1936, 0.2590])
            Loss["SubNet+R-MUSIC"]  = np.array([0.0208, 0.0335, 0.0397, 0.0453, 0.0664, 0.0890])
            Loss["MUSIC"]           = np.array([0.0649, 0.1027, 0.1307, 0.1510, 0.1881, 0.2322])
            Loss["SubNet+MUSIC"]    = np.array([0.0564, 0.0707, 0.0796, 0.0840, 0.1123, 0.1402])
            Loss["ESPRIT"]          = np.array([0.0422, 0.0749, 0.1067, 0.1437, 0.2008, 0.2543])
            Loss["SubNet+ESPRIT"]   = np.array([0.0261, 0.0405, 0.0497, 0.0573, 0.0766, 0.0939])
    
    elif simulation.startswith("sv_calibration"):
        x_axis["psi"] = [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075]
        if RMSPE:
            Loss["R-MUSIC"]         = np.array([0.0398, 0.0741, 0.1030, 0.1410, 0.1936, 0.2590])
            Loss["SubNet+R-MUSIC"]  = np.array([0.0208, 0.0335, 0.0397, 0.0453, 0.0664, 0.0890])
            Loss["MUSIC"]           = np.array([0.0649, 0.1027, 0.1307, 0.1510, 0.1881, 0.2322])
            Loss["SubNet+MUSIC"]    = np.array([0.0564, 0.0707, 0.0796, 0.0840, 0.1123, 0.1402])
            Loss["ESPRIT"]          = np.array([0.0422, 0.0749, 0.1067, 0.1437, 0.2008, 0.2543])
            Loss["SubNet+ESPRIT"]   = np.array([0.0261, 0.0405, 0.0497, 0.0573, 0.0766, 0.0939])

    return x_axis, Loss
def plot(x_axis, Loss, conv_method):
    notations = {"MUSIC"            : {"linestyle":'solid',  "marker":'D', "color":'#104E8B'},
                 "SPS+MUSIC"        : {"linestyle":'solid',  "marker":'x', "color":'#CD6600'},
                 "R-MUSIC"          : {"linestyle":'solid',  "marker":'s', "color":'#006400'},
                 "SPS+R-MUSIC"      : {"linestyle":'solid',  "marker":'*', "color":'#FFA500'},
                 "ESPRIT"           : {"linestyle":'solid',  "marker":'o', "color":'#BF3EFF'},
                 "SubNet+MUSIC"     : {"linestyle":'dashed', "marker":'<', "color":'#104E8B'},
                 "SubNet+R-MUSIC"   : {"linestyle":'dashed', "marker":'>', "color":'#006400'},
                 "SubNet+ESPRIT"    : {"linestyle":'dashed', "marker":'P', "color":'#BF3EFF'}}
    
    fig = plt.figure(figsize=(8, 6))
    plt.style.use('plot_style.txt')
    for axis_name, axis_array in x_axis.items():
        axis_name = axis_name
        axis_array = axis_array
        
    for method, loss in Loss.items():
        plt.plot(axis_array, conv_method(loss), linestyle = notations[method]["linestyle"],
                 marker=notations[method]["marker"], label=method, color=notations[method]["color"])

    # plt.xlim([0.01, 0.08])
    # plt.ylim([-45, -15])
    plt.xlabel(axis_name)
    plt.ylabel("RMSPE [rad]")
    plt.legend()

if __name__ == "__main__":
    x_axis, Loss = scenario_to_plot(simulation="distance_calibration")
    plot(x_axis, Loss, unit)
    plt.show()

# plt.xlim([4.9, 10.1])
# plt.ylim([2.5* 1e-2, 5.8 * 1e-1])
# plt.xscale('log')
# plt.ylabel("RMSPE [rad]")
# plt.yscale("log", base=10)
# plt.grid(which='both')
# plt.grid(which='both')

# Add legend to plot
# plt.legend(bbox_to_anchor=(0.41, 0.34), loc=0)

# # coherent sources, T = 200
# elif SNR_T_200_COHERENT:
#     ## RMSPE Scores

# ##################################
# ### SNR  simulation for T = 20 ###
# ##################################
# # coherent sources, T = 20
# elif SNR_T_20_COHERENT:
#     ## RMSPE Scores
#     if RMSPE:
#         MUSIC_coherent              = np.array([0.174, 0.171, 0.170, 0.169, 0.169, 0.169])
#         RootMUSIC_coherent          = np.array([0.218, 0.216, 0.216, 0.216, 0.217, 0.217])
#         SPS_MUSIC_coherent          = np.array([0.067, 0.0598, 0.052, 0.047, 0.042, 0.04])
#         SPS_RootMUSIC_coherent      = np.array([0.049, 0.043, 0.037, 0.0323, 0.027, 0.024])
#         DeepRootMUSIC_coherent      = np.array([0.0088, 0.0075, 0.0075, 0.0078, 0.0066, 0.006])
    
#     ## MSPE Scores
#     else:
#         MUSIC_coherent              =  np.array([0.0935, 0.0917, 0.0903, 0.0898, 0.0900, 0.0856])
#         RootMUSIC_coherent          =  np.array([0.1262, 0.1245, 0.1243, 0.1245, 0.1253, 0.1228])
#         SPS_RootMUSIC_coherent      =  np.array([0.0259, 0.0234, 0.0194, 0.0167, 0.0143, 0.0120])
#         SPS_MUSIC_coherent          =  np.array([0.0428, 0.0383, 0.0336, 0.0305, 0.0283, 0.0267])
#         DeepRootMUSIC_coherent      =  np.array([0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0000])

##################################
#### SNR simulation for T = 2 ####
##################################
# Non-coherent sources, T = 2
# elif SNR_T_2_NON_COHERENT:
#     snr        = np.array([-5, -4, -3, -2, -1])
#     ## RMSPE Scores
#     if RMSPE:
#         MUSIC_non_coherent          = np.array([0.424, 0.423, 0.416, 0.423, 0.412, 0.412])
#         RootMUSIC_non_coherent      = np.array([0.479, 0.471, 0.465, 0.468, 0.458, 0.4609])
#         SPS_RootMUSIC_non_coherent  = np.array([0.156, 0.132, 0.115, 0.094, 0.08,0.07])
#         SPS_MUSIC_non_coherent      = np.array([0.1827, 0.161, 0.141, 0.124, 0.11, 0.098])
#         DeepRootMUSIC_non_coherent  = np.array([0.05, 0.0418, 0.035, 0.0298, 0.0271, 0.024])
#     ## MSPE Scores
#     else:
#         MUSIC_non_coherent          = np.array([0.2697, 0.2694, 0.2635, 0.2708, 0.2584, 0.2610])
#         RootMUSIC_non_coherent      = np.array([0.3189, 0.3117, 0.3083, 0.3089, 0.3002, 0.3045])
#         SPS_MUSIC_non_coherent      = np.array([0.1101, 0.0981, 0.0861, 0.0765, 0.0687, 0.0614])
#         SPS_RootMUSIC_non_coherent  = np.array([0.0848, 0.0714, 0.0582, 0.0484, 0.0407, 0.0346])
#         DeepRootMUSIC_non_coherent  = np.array([0.0117, 0.0087, 0.0064, 0.0045, 0.0035, 0.0032])
    
# # coherent sources, T = 200
# elif SNR_T_2_COHERENT:
#     ## RMSPE Scores
#     if RMSPE:
#         MUSIC_coherent              = np.array([0.383, 0.354, 0.348, 0.322, 0.325, 0.314])
#         RootMUSIC_coherent          = np.array([0.46, 0.447, 0.449, 0.432, 0.437, 0.43])
#         SPS_MUSIC_coherent          = np.array([0.1657, 0.141, 0.125, 0.11, 0.1, 0.088])
#         SPS_RootMUSIC_coherent      = np.array([0.1444, 0.1196, 0.1025, 0.087, 0.075,0.065])
#         DeepRootMUSIC_coherent      = np.array([0.033, 0.027, 0.024, 0.02, 0.017, 0.015])    

#     ## MSPE Scores
#     else:
#         MUSIC_coherent              = np.array([0.2471, 0.2231, 0.2196, 0.1980, 0.2023, 0.1894])
#         RootMUSIC_coherent          = np.array([0.3179, 0.3027, 0.3064, 0.2949, 0.2988, 0.2917])
#         SPS_MUSIC_coherent          = np.array([0.0972, 0.0815, 0.0734, 0.0653, 0.0603, 0.0534])
#         SPS_RootMUSIC_coherent      = np.array([0.0767, 0.0609, 0.0513, 0.0422, 0.0362, 0.0316])
#         DeepRootMUSIC_coherent      = np.array([0.0068, 0.0049, 0.0040, 0.0036, 0.0031, 0.0027])