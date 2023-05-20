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
from numpy import linalg as LA

######################
# Conversion methods #
######################

def unit(value):
    return value

def rad2deg(value):
    return value * 180 / np.pi

def deg2rad(value):
    return value * np.pi / 180

def rad2dB(value:float):
    """Converts MSE value in radian scale into dB scale

    Args:
        mse_value (float): value of error in rad

    Returns:
        float: value of error in dB
    """    
    return 10 * np.log10(value)


def scenario_to_plot(simulation, conv_method=unit, mode = "non-coherent", T = 200):
    Loss = {}
    x_axis = {}
    
    # Criterion
    RMSPE = True
    if conv_method == rad2dB:
        RMSPE = False
    
    ##### SNR simulation #####
    if simulation.startswith("SNR"):
        # T = 200
        if T == 200:
            x_axis["SNR"] = np.array([-5, -4, -3, -2, -1])
            # non-coherent sources
            if mode.startswith("non-coherent"):            
                # RMSPE Scores
                if RMSPE:
                    Loss["MUSIC"]           = np.array([0.097, 0.0773, 0.0626, 0.0499, 0.04])
                    Loss["R-MUSIC"]         = np.array([0.073, 0.055, 0.0409, 0.0304, 0.022])
                    Loss["SubNet+R-MUSIC"]  = np.array([0.0399, 0.0297, 0.0228, 0.0164, 0.0133])  
                # MSPE Scores
                else:
                    Loss["SubNet+R-MUSIC"]  = np.array([0.0040, 0.0032, 0.0014, 0.0007, 0.0005])
                    Loss["R-MUSIC"]         = np.array([0.0312, 0.0214, 0.0148, 0.0097, 0.0062])

                    Loss["SubNet+ESPRIT"]   = np.array([0.007, 0.0035, 0.00235, 0.001938, 0.001636])
                    Loss["ESPRIT"]          = np.array([0.0244, 0.0159, 0.0101, 0.0073, 0.00426])
                    # uniform in [-90, 90]
                    # Loss["SubNet+MUSIC"]    = np.array([0.0342, 0.0265, 0.0172, 0.0179, 0.01991])
                    # Loss["MUSIC"]           = np.array([0.0503, 0.0400, 0.0321, 0.0302, 0.0296])
                    
                    # uniform in [-80, 80]    
                    Loss["SubNet+MUSIC"]    = np.array([0.0052, 0.005, 0.00486, 0.00462, 0.004])
                    Loss["MUSIC"]           = np.array([0.0221, 0.01305, 0.0085, 0.00761, 0.00701])
            
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
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0031, 0.0011, 0.0007, 0.0005, 0.0003])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0829, 0.0619, 0.0536, 0.0443, 0.0394])
                    Loss["R-MUSIC"]          = np.array([0.1441, 0.1369, 0.1334, 0.1279, 0.1258])
                    
                    Loss["SubNet+ESPRIT"]    = np.array([0.01475, 0.01232, 0.01127, 0.00683, 0.0049])
                    Loss["SPS+ESPRIT"]       = np.array([0.0806, 0.06516, 0.0506, 0.03955, 0.0347])
                    Loss["ESPRIT"]           = np.array([0.2785, 0.278419, 0.2718, 0.2774, 0.27714])
                    
                    # Run in higher gap
                    # Loss["SubNet+MUSIC"]     = np.array([0.0534, 0.0506, 0.0460, 0.03298, 0.01295])
                    # Loss["SPS+MUSIC"]        = np.array([0.0942, 0.0723, 0.0638, 0.0569, 0.054])
                    # Loss["MUSIC"]            = np.array([0.1124, 0.1055, 0.0954, 0.0976, 0.0948])
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
                    Loss["SubNet+R-MUSIC"]   = np.array([0.000285, 0.000216, 0.000214, 0.000126, 0.000101, 0.000097])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0428, 0.0383, 0.0336, 0.0305, 0.0283, 0.0267])
                    Loss["R-MUSIC"]          = np.array([0.1262, 0.1245, 0.1243, 0.1245, 0.1253, 0.1228])
                    
                    Loss["SubNet+ESPRIT"]    = np.array([0.00693, 0.00701, 0.007167, 0.00580, 0.00616, 0.006049])
                    Loss["SPS+ESPRIT"]       = np.array([0.022, 0.0196, 0.0165, 0.0139, 0.01152, 0.00998])
                    Loss["ESPRIT"]           = np.array([0.27377, 0.27838, 0.278311, 0.27821, 0.27817, 0.27119])
                    
                    # Loss["SubNet+MUSIC"]     = np.array([0.0473, 0.0499, 0.0499, 0.03234, 0.03349, 0.0279])
                    # Loss["SPS+MUSIC"]        = np.array([0.0259, 0.0234, 0.0194, 0.0167, 0.0143, 0.0120])
                    # Loss["MUSIC"]            = np.array([0.0935, 0.0917, 0.0903, 0.0898, 0.0900, 0.087])
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
                    Loss["SubNet+R-MUSIC"]  = np.array([0.0117, 0.0087, 0.0064, 0.0045, 0.0035, 0.0032])
                    Loss["SubNet+MUSIC"]    = np.array([0.039, 0.033, 0.03064, 0.0287, 0.028, 0.0267])
                    Loss["SubNet+ESPRIT"]   = np.array([0.0119, 0.009, 0.0078, 0.00573, 0.00564, 0.00478])
                    
                    Loss["R-MUSIC"]         = np.array([0.3189, 0.3117, 0.3083, 0.3089, 0.3002, 0.3045])
                    Loss["MUSIC"]           = np.array([0.2697, 0.2694, 0.2635, 0.2708, 0.2584, 0.2610])
                    Loss["ESPRIT"]          = np.array([0.3219, 0.3267, 0.32287, 0.32017, 0.31894, 0.32557])
                    
                    Loss["SPS+R-MUSIC"]     = np.array([0.0848, 0.0714, 0.0582, 0.0484, 0.0407, 0.0346])
                    Loss["SPS+MUSIC"]       = np.array([0.1101, 0.0981, 0.0861, 0.0765, 0.0687, 0.0614])
                    Loss["SPS+ESPRIT"]      = np.array([0.0775, 0.06479, 0.05335, 0.0439, 0.0358, 0.0293])
                    
            
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
                    Loss["SubNet+R-MUSIC"]   = np.array([0.0068, 0.0049, 0.0040, 0.0036, 0.0031, 0.0027])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0767, 0.0609, 0.0513, 0.0422, 0.0362, 0.0316])
                    Loss["R-MUSIC"]          = np.array([0.3179, 0.3027, 0.3064, 0.2949, 0.2988, 0.2917])
                    # Loss["SubNet+MUSIC"]     = np.array([0.0702, 0.0675, 0.06051, 0.05604, 0.05340, 0.0506472])
                    Loss["SubNet+ESPRIT"]    = np.array([0.01102, 0.00935, 0.00829, 0.00782, 0.00751, 0.0073])
                    Loss["SPS+ESPRIT"]       = np.array([0.0683, 0.0545, 0.0451, 0.0385, 0.0327, 0.027654])
                    # Loss["MUSIC"]            = np.array([0.2471, 0.2231, 0.2196, 0.1980, 0.2023, 0.1894])
                    Loss["ESPRIT"]           = np.array([0.32363, 0.32217, 0.327, 0.3207, 0.32317, 0.327887])
                    # Loss["SPS+MUSIC"]        = np.array([0.0972, 0.0815, 0.0789, 0.0653, 0.0603, 0.0534])
    
    ###### BroadBand simulation  #####
    elif simulation.startswith("OFDM"):
        x_axis["Observations"] = [50, 100, 200, 500, 1000]
        if mode.startswith("non-coherent"):            
            if RMSPE:
                Loss["SubNet+R-MUSIC"]  = np.array([0.088 , 0.056 ,0.0517, 0.0428, 0.0239])
                Loss["SubNet+MUSIC"]    = np.array([0.1221, 0.09302, 0.07762, 0.06227, 0.05966])
                Loss["SubNet+ESPRIT"]   = np.array([0.086 , 0.0604 , 0.05605 ,0.0466, 0.0436])
                Loss["BB-MUSIC"]        = np.array([0.22733, 0.18433 ,0.1549, 0.125, 0.121])
                Loss["MUSIC"]           = np.array([0.5207, 0.5297, 0.533782,0.5572, 0.5687])
                Loss["ESPRIT"]          = np.array([0.533725, 0.559649, 0.5805,0.6094, 0.6131])
                Loss["R-MUSIC"]         = np.array([0.537, 0.54728, 0.55966, 0.59487, 0.600])
                # Loss["DA-MUSIC"]        = np.array([0.192, 0.125, 0.073 ,0.074])
        elif mode.startswith("coherent"):            
            if RMSPE:
                Loss["SubNet+R-MUSIC"]  = np.array([0.0704 ,0.0520 ,0.0445 ,0.0284, 0.0268])
                Loss["SubNet+MUSIC"]    = np.array([0.0974 ,0.0938 ,0.0916 ,0.0868, 0.0845])
                Loss["SubNet+ESPRIT"]   = np.array([0.0781 ,0.0671 ,0.0600 ,0.0454, 0.0444])
                Loss["BB-MUSIC"]        = np.array([0.2417 ,0.2078 ,0.1836 ,0.2104, 0.2802])
                # Loss["MUSIC"]           = np.array([0.5176 ,0.5118 ,0.5077 ,0.5069, 0.5088])
                # Loss["ESPRIT"]          = np.array([0.5478 ,0.5714 ,0.6031 ,0.6307, 0.6582])
                # Loss["R-MUSIC"]         = np.array([0.5370 ,0.5399 ,0.5569 ,0.5746, 0.6019])
                # Loss["DA-MUSIC"]        = np.array([0.192, 0.125, 0.073 ,0.074])
        # Add here music and hybrid music

    ###### mis-calibration simulations  ######
    elif simulation.startswith("distance_calibration"):
        x_axis["eta"] = [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075]
        if RMSPE:
            Loss["SubNet+R-MUSIC"]  = np.array([0.0208, 0.0335, 0.0397, 0.0453, 0.0664, 0.0890])
            Loss["SubNet+MUSIC"]    = np.array([0.0564, 0.0707, 0.0796, 0.0840, 0.0912, 0.1012])
            Loss["SubNet+ESPRIT"]   = np.array([0.0261, 0.0405, 0.0497, 0.0573, 0.0766, 0.0939])
            Loss["MUSIC"]           = np.array([0.0649, 0.1027, 0.1307, 0.1510, 0.1881, 0.2322])
            Loss["R-MUSIC"]         = np.array([0.0398, 0.0741, 0.1030, 0.1410, 0.1936, 0.2590])
            Loss["ESPRIT"]          = np.array([0.0422, 0.0749, 0.1067, 0.1437, 0.2008, 0.2543])
    
    elif simulation.startswith("sv_noise"):
        x_axis["sigma"] = [0.075, 0.1, 0.3, 0.4, 0.5, 0.75]
        if RMSPE:
            Loss["SubNet+R-MUSIC"]  = np.array([0.0148, 0.0173, 0.0304, 0.0408, 0.0506, 0.0807])
            Loss["SubNet+MUSIC"]    = np.array([0.0447, 0.0476, 0.0688, 0.0863, 0.0905, 0.1051])
            Loss["SubNet+ESPRIT"]   = np.array([0.0236, 0.0246, 0.0409, 0.0525, 0.0626, 0.0930])
            Loss["MUSIC"]           = np.array([0.0687, 0.0750, 0.1042, 0.1159, 0.1259, 0.1840])
            Loss["R-MUSIC"]         = np.array([0.0353, 0.0485, 0.0750, 0.0984, 0.1221, 0.1803])
            Loss["ESPRIT"]          = np.array([0.0370, 0.0455, 0.0860, 0.1112, 0.1323, 0.1905])
    return x_axis, Loss

def plot(x_axis, Loss, conv_method, algorithm="all"):
    notations = {"MUSIC"            : {"linestyle":'solid',  "marker":'D', "color":'#104E8B'},
                 "SPS+MUSIC"        : {"linestyle":'dashdot',  "marker":'x', "color":'#0f83f5'},
                 "SubNet+MUSIC"     : {"linestyle":'dashed', "marker":'<', "color":'#1a05e3'},
                 "R-MUSIC"          : {"linestyle":'solid',  "marker":'s', "color":'#006400'},
                 "SPS+R-MUSIC"      : {"linestyle":'dashdot',  "marker":'*', "color":'#0d8074'},
                 "SubNet+R-MUSIC"   : {"linestyle":'dashed', "marker":'>', "color":'#039403'},
                 "ESPRIT"           : {"linestyle":'solid',  "marker":'o', "color":'#842ab0'},
                 "SPS+ESPRIT"       : {"linestyle":'dashdot',  "marker":'*', "color":'#9f59c2'},
                 "SubNet+ESPRIT"    : {"linestyle":'dashed', "marker":'P', "color":'#BF3EFF'},
                 "BB-MUSIC"         : {"linestyle":'solid', "marker":'x', "color":'#FFA500'}}
    
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    plt.style.use('plot_style.txt')
    for axis_name, axis_array in x_axis.items():
        axis_name = axis_name
        axis_array = axis_array
        
    for method, loss in Loss.items():
        if algorithm == "all" or algorithm in method:
            if algorithm == "MUSIC" and "R-MUSIC" in method:
                pass
            else:
                plt.plot(axis_array, conv_method(loss), linestyle = notations[method]["linestyle"],
                        marker=notations[method]["marker"], label=method, color=notations[method]["color"])
        elif algorithm == "coherent":
            if "SPS" in method or "SubNet" in method:
                plt.plot(axis_array, conv_method(loss), linestyle = notations[method]["linestyle"],
                        marker=notations[method]["marker"], label=method, color=notations[method]["color"])
        else:
            pass
            
    
    plt.xlim([np.min(axis_array) - 0.1 * np.abs(np.min(axis_array)), np.max(axis_array) + 0.1 * np.abs(np.min(axis_array))])
    # plt.ylim([-45, -15])
    plt.xlabel(axis_name)
    if conv_method == unit:
        plt.ylabel("RMSPE [rad]")
    elif conv_method == rad2dB:
        plt.ylabel("MSPE [dB]")
    plt.legend()

if __name__ == "__main__":
    ########################################################
    # simulation = "SNR"
    # conv_method = rad2dB
    # mode = "non-coherent"
    # T = 200
    # algorithm="all"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=conv_method, mode=mode, T=T)
    # plot(x_axis, Loss, conv_method=conv_method, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -14])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    # ########################################################
    # scenario_to_plot(simulation="distance_calibration")
    # T = 20
    # mode = "non-coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode=mode, T = T)

    ## ESPRIT
    # algorithm="ESPRIT"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.ylim([-24, -5])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    # plt.xlim([4.9, 10.1])
    
    ## MUSIC
    # algorithm="MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-20, -9])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    ## R-MUSIC
    # algorithm="R-MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -6])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')

    # ## Coherent
    # algorithm="coherent"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -10])
    # # plt.legend(bbox_to_anchor=(0.41, 0.34), loc=0)
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')

    ## All
    # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.xscale("log", base=10)
    # plt.ylim([0.015, 0.63])
    # plt.ylim([0.015, 0.23])
    # plt.xlim([47, 1050])
    # plt.legend(fontsize='x-small', bbox_to_anchor=(0.21, 0.57))
    # plt.legend(fontsize='small', bbox_to_anchor=(0.21, 0.5))
    # plt.legend(bbox_to_anchor=(0.41, 0.34))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    # plt.savefig("{}_{}_{}_no_classical_methods.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode=mode, T = T)
    # ## ESPRIT
    # algorithm="ESPRIT"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-24, -4])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    ## distance_calibration
    # plt.xlim([4.9, 10.1])
    
    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 200
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -5])
    # # plt.legend(bbox_to_anchor=(0.21, 0.34), loc=0)
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################

    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 20
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-41, -5])
    # plt.legend(bbox_to_anchor=(0.41, 0.54), loc=0)
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################
    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 2
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-26, -4.5])
    # # plt.legend(bbox_to_anchor=(0.41, 0.54), loc=0)
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################

    # ########################################################
    # simulation = "distance_calibration"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.ylim([0.015, 0.27])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    # ########################################################

    # ########################################################
    # mode = "coherent"
    # simulation = "OFDM"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.xlim([47, 1050])
    # plt.xscale("log", base=10)
    # # plt.ylim([0.02, 0.68])
    # plt.ylim([0.02, 0.29])
    # plt.savefig("{}_{}_no_narrowband.pdf".format(simulation, algorithm),bbox_inches='tight')
    # ########################################################

    ########################################################
    simulation = "sv_noise"
    x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    algorithm="all"
    plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    plt.xlabel(r"$\sigma^2_{\rm sv}$")
    plt.xlim([0.065, 0.76])
    # plt.xscale("log", base=10)
    # plt.ylim([0.02, 0.68])
    plt.ylim([0.01, 0.195])
    plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    ########################################################
    
    
    
    
    
    # algorithm="MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-16, -5])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    # ## R-MUSIC
    # algorithm="R-MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-26, -4])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    # plt.yticks(np.arange(-35, -14.5, 2.5))
    # scenario_to_plot(simulation="SNR", conv_method=rad2dB, mode="non-coherent", T = 2)
    # scenario_to_plot(simulation="SNR", conv_method=rad2dB, mode="coherent", T = 2)
    # plt.show()

    #######################################################
    # simulation = "eigenvalues"
    empirical_cov = [[   1.32462396  +0.j        ,    1.37029507 +12.71751673j,
           3.19058897 +11.36444576j,   -3.62373953  +0.217819j  ,
         -14.07081456  -1.83435012j,   -9.18583814  +1.72975747j,
           0.35775326  -3.06552153j,    1.92934983 -12.16284328j],
       [   1.37029507 -12.71751673j,  123.51651923  +0.j        ,
         112.40894199 -18.87609266j,   -1.65743318 +35.01631097j,
         -32.1672774 +133.19419286j,    7.10459059 +89.98125679j,
         -29.06152625  -6.60595185j, -114.77777044 -31.10560002j],
       [   3.19058897 -11.36444576j,  112.40894199 +18.87609266j,
         105.18493554  +0.j        ,   -6.85966093 +31.61407584j,
         -49.62960083+116.30044207j,   -7.28546299 +82.97517489j,
         -25.43854764 -10.45314528j,  -99.70241736 -45.84895571j],
       [  -3.62373953  -0.217819j  ,   -1.65743318 -35.01631097j,
          -6.85966093 -31.61407584j,    9.94918839  +0.j        ,
          38.19152628  +7.33196595j,   25.41389852  -3.22154864j,
          -1.48278568  +8.32743969j,   -7.27811052 +32.95631695j],
       [ -14.07081456  +1.83435012j,  -32.1672774 -133.19419286j,
         -49.62960083-116.30044207j,   38.19152628  -7.33196595j,
         152.00741458  +0.j        ,   95.18115963 -31.09496836j,
           0.44492634 +33.05891425j,   -3.65130843+131.87155093j],
       [  -9.18583814  -1.72975747j,    7.10459059 -89.98125679j,
          -7.28546299 -82.97517489j,   25.41389852  +3.22154864j,
          95.18115963 +31.09496836j,   65.95961278  +0.j        ,
          -6.48400797 +20.79122767j,  -29.26224018 +81.82594155j],
       [   0.35775326  +3.06552153j,  -29.06152625  +6.60595185j,
         -25.43854764 +10.45314528j,   -1.48278568  -8.32743969j,
           0.44492634 -33.05891425j,   -6.48400797 -20.79122767j,
           7.19102929  +0.j        ,   28.66903396  +1.18008334j],
       [   1.92934983 +12.16284328j, -114.77777044 +31.10560002j,
         -99.70241736 +45.84895571j,   -7.27811052 -32.95631695j,
          -3.65130843-131.87155093j,  -29.26224018 -81.82594155j,
          28.66903396  -1.18008334j,  114.49071775  +0.j        ]]
    ssn_cov = [[  56731.0547+0.0000j,   -4314.5723-98431.1094j,
         -123549.7266+117745.7031j,   25086.9844+128895.8984j,
           64204.0430+39669.5938j, -111450.4844-118976.4844j,
         -100929.3750+28066.1016j,   30292.9414+13613.3086j],
        [  -4314.5723+98431.1094j,  833051.8750+0.0000j,
           29494.4316-299663.6250j, -180515.4688+362784.9375j,
          -60003.3320+566930.3125j,  316108.0312-2534.6602j,
         -278758.6250-286772.8438j,  -99148.2031+141427.1719j],
        [-123549.7266-117745.7031j,   29494.4316+299663.6250j,
          892673.0625+0.0000j,  123809.5547-337977.7188j,
         -187054.2188+115169.7891j,   56510.7383+824029.7500j,
          347948.0000-10480.0703j, -194997.6719-171329.5938j],
        [  25086.9844-128895.8984j, -180515.4688-362784.9375j,
          123809.5547+337977.7188j,  857850.5625+0.0000j,
          386227.1875-195151.2500j, -301688.6562+117594.9141j,
           -5905.2197+697171.1875j,  175821.2812+23194.7324j],
        [  64204.0430-39669.5938j,  -60003.3320-566930.3125j,
         -187054.2188-115169.7891j,  386227.1875+195151.2500j,
          683439.9375+0.0000j,   90717.5000-249569.8125j,
         -225344.0312+242898.7188j,   54673.5938+239450.0469j],
        [-111450.4844+118976.4844j,  316108.0312+2534.6602j,
           56510.7383-824029.7500j, -301688.6562-117594.9141j,
           90717.5000+249569.8125j,  944404.6875+0.0000j,
           44709.7578-305063.6250j, -139964.2344+200239.0469j],
        [-100929.3750-28066.1016j, -278758.6250+286772.8438j,
          347948.0000+10480.0703j,   -5905.2197-697171.1875j,
         -225344.0312-242898.7188j,   44709.7578+305063.6250j,
          710608.1250+0.0000j,  -40602.5508-164594.3125j],
        [  30292.9414-13613.3086j,  -99148.2031-141427.1719j,
         -194997.6719+171329.5938j,  175821.2812-23194.7324j,
           54673.5938-239450.0469j, -139964.2344-200239.0469j,
          -40602.5508+164594.3125j,  226326.3438+0.0000j]]
    sps_cov = [[ 59.99381678 +0.j        ,  36.2777756  +8.19686646j,
         -5.67063663+39.86491254j, -11.1398164 +56.17865661j,
         -9.92072053+27.66251958j],
       [ 36.2777756  -8.19686646j,  97.66451444 +0.j        ,
         59.73049174 -2.75625481j,  -6.35705229+45.28852966j,
        -11.14670862+89.09208959j],
       [ -5.67063663-39.86491254j,  59.73049174 +2.75625481j,
         83.27528782 +0.j        ,  30.00725425 +7.16057528j,
        -13.25825404+56.99093731j],
       [-11.1398164 -56.17865661j,  -6.35705229-45.28852966j,
         30.00725425 -7.16057528j,  58.77681126 +0.j        ,
         38.88942797 -0.44792285j],
       [ -9.92072053-27.66251958j, -11.14670862-89.09208959j,
        -13.25825404-56.99093731j,  38.88942797 +0.44792285j,
         84.9121936  +0.j        ]]
    # Deep_RootMUSIC_Rx = [[[162162.7500+0.0000j, -53419.5820-29567.0664j,
    # 72869.0625+45681.9219j,  18064.2520-29616.1172j,
    # 138235.3750-11426.2324j, -69004.9141-33662.0234j,
    # 101304.8906+35886.5547j, -31843.0645-50518.2773j],
    # [-53419.5820+29567.0664j, 130456.1172+0.0000j,
    # -38763.6680-52971.1641j,  78656.8203+35861.0977j,
    # -21711.6445-38134.4062j,  56236.5742-67059.6250j,
    # -74090.4531+10358.3154j,  61832.4805+10086.8291j],
    # [ 72869.0625-45681.9219j, -38763.6680+52971.1641j,
    # 172605.5938+0.0000j,  -3078.1484-31445.6250j,
    # 114771.0078-7304.6855j,  57420.8672-17480.7793j,
    # 67166.2031-75923.6250j, -69415.0156+20219.1660j],
    # [ 18064.2520+29616.1172j,  78656.8203-35861.0977j,
    # -3078.1484+31445.6250j, 123215.0781+0.0000j,
    # 8936.6953-19623.5977j,  31236.8535-37875.9219j,
    # 20523.3242+31001.5547j,  24692.6172-31873.0938j],
    # [138235.3750+11426.2324j, -21711.6445+38134.4062j,
    # 114771.0078+7304.6855j,   8936.6953+19623.5977j,
    # 193280.2812+0.0000j,   -224.4319-61926.8867j,
    # 60970.2070+7851.4541j, -16777.9082-12268.1699j],
    # [-69004.9141+33662.0234j,  56236.5742+67059.6250j,
    # 57420.8672+17480.7793j,  31236.8535+37875.9219j,
    # -224.4319+61926.8867j, 150367.0000+0.0000j,
    # -41780.7305-50633.0234j,  10084.9014+38179.3789j],
    # [101304.8906-35886.5547j, -74090.4531-10358.3154j,
    # 67166.2031+75923.6250j,  20523.3242-31001.5547j,
    # 60970.2070-7851.4541j, -41780.7305+50633.0234j,
    # 134363.7500+0.0000j, -69894.4453-33309.3672j],
    # [-31843.0645+50518.2773j,  61832.4805-10086.8291j,
    # -69415.0156-20219.1660j,  24692.6172+31873.0938j,
    # -16777.9082+12268.1699j,  10084.9014-38179.3789j,
    # -69894.4453+33309.3672j,  78712.8281+0.0000j]]]

    # M = 3
    # empirical_eig = np.sort(np.real(LA.eigvals(empirical_cov)))[::-1]
    # norm_empirical_eig = empirical_eig / np.max(empirical_eig)
    
    # sps_eig = np.sort(np.real(LA.eigvals(sps_cov)))[::-1]
    # norm_sps_eig = sps_eig / np.max(sps_eig)
    
    # ssn_eig = np.sort(np.real(LA.eigvals(ssn_cov)))[::-1]
    # norm_ssn_eig = ssn_eig / np.max(ssn_eig)
    
    # algorithm = "ssn"
    # plt.style.use('default')
    # fig = plt.figure(figsize=(7, 5.5))
    # plt.style.use('plot_style.txt')

    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalues [λ]")
    # plt.xlim([0.85, 8.15])
    # plt.ylim([-0.02, 1.02])
    
    # markerline, stemlines, baseline = plt.stem([i + 1 + 0.05 for i in range(empirical_eig.shape[0])],norm_empirical_eig, '#842ab0', label="Empirical")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.setp(stemlines, 'linestyle', 'dashed')
    # markerline, stemlines, baseline = plt.stem([i + 1 + 0.05 for i in range(norm_sps_eig.shape[0])],norm_sps_eig, '#0f83f5', label="SPS")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.setp(stemlines, 'linestyle', 'dashed')
    # markerline, stemlines, baseline = plt.stem([i + 1 - 0.05 for i in range(norm_ssn_eig.shape[0])], norm_ssn_eig,'#039403', markerfmt='>', label="SubNet")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.legend()
    # plt.savefig("eigenvalues.pdf",bbox_inches='tight')

    # # markerline, stemlines, baseline = plt.stem(x, y, markerfmt='o', label='pcd')
    # # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # # plt.setp(stemlines, 'linestyle', 'dotted')

    # algorithm = "rm"
    # plt.style.use('default')
    # fig = plt.figure(figsize=(7, 5.5))
    # plt.style.use('plot_style.txt')

    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalues [λ]")
    # plt.xlim([0.9, 8.1])
    # plt.ylim([-0.02, 1.02])
    # plt.ylim([-10000, 6 * 100000])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    
    
    plt.show()
    