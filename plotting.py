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
                    # Loss["SubNet+R-MUSIC"]   = np.array([0.0031, 0.0011, 0.0007, 0.0005, 0.0003])
                    # Loss["SPS+R-MUSIC"]      = np.array([0.0829, 0.0619, 0.0536, 0.0443, 0.0394])
                    # Loss["R-MUSIC"]          = np.array([0.1441, 0.1369, 0.1334, 0.1279, 0.1258])
                    
                    Loss["SubNet+ESPRIT"]    = np.array([0.01475, 0.01232, 0.01127, 0.00683, 0.0049])
                    Loss["SPS+ESPRIT"]       = np.array([0.0806, 0.06516, 0.0506, 0.03955, 0.0347])
                    Loss["ESPRIT"]           = np.array([0.2785, 0.278419, 0.2718, 0.2774, 0.27714])
                    
                    # Run in higher gap
                    Loss["SubNet+MUSIC"]     = np.array([0.0534, 0.0506, 0.0460, 0.03298, 0.01295])
                    Loss["SPS+MUSIC"]        = np.array([0.0942, 0.0723, 0.0638, 0.0569, 0.054])
                    Loss["MUSIC"]            = np.array([0.1124, 0.1055, 0.0954, 0.0976, 0.0948])
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
                    # Loss["SubNet+MUSIC"]     = np.array([0.0702, 0.0675, 0.06051, 0.05604, 0.05340, 0.0506472])
                    Loss["SubNet+ESPRIT"]    = np.array([0.01102, 0.00935, 0.00829, 0.00782, 0.00751, 0.0073])
                    # Loss["MUSIC"]            = np.array([0.2471, 0.2231, 0.2196, 0.1980, 0.2023, 0.1894])
                    Loss["R-MUSIC"]          = np.array([0.3179, 0.3027, 0.3064, 0.2949, 0.2988, 0.2917])
                    Loss["ESPRIT"]           = np.array([0.32363, 0.32217, 0.327, 0.3207, 0.32317, 0.327887])
                    Loss["SPS+R-MUSIC"]      = np.array([0.0767, 0.0609, 0.0513, 0.0422, 0.0362, 0.0316])
                    # Loss["SPS+MUSIC"]        = np.array([0.0972, 0.0815, 0.0789, 0.0653, 0.0603, 0.0534])
                    Loss["SPS+ESPRIT"]       = np.array([0.0683, 0.0545, 0.0451, 0.0385, 0.0327, 0.027654])
    
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
    
    # ########################################################
    simulation = "distance_calibration"
    x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    algorithm="all"
    plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    plt.xlabel(r"$\eta [\lambda / 2]$")
    plt.ylim([0.015, 0.27])
    plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
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
    # simulation = "sv_noise"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\sigma^2$")
    # plt.xlim([0.065, 0.76])
    # # plt.xscale("log", base=10)
    # # plt.ylim([0.02, 0.68])
    # plt.ylim([0.01, 0.195])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
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
    # RootMUSIC_Rx = [[ 1.73716980e+03+7.64657289e-15j, -2.39444190e+02-6.56316090e+02j,
    # 7.23856406e+02+2.57973205e+02j,  4.69720262e+02-1.69492130e+02j,
    # 9.53657842e+02-8.88213300e+02j, -7.24412394e+02-2.70281966e+02j,
    # 1.28687308e+03-5.90436586e+01j, -2.62877100e+02-1.24084339e+03j],
    # [-2.39444190e+02+6.56316090e+02j,  2.83417070e+02+5.02114423e-16j,
    # -1.97789098e+02+2.38177650e+02j, -6.17562632e-01+2.01077410e+02j,
    # 2.04433876e+02+4.83299448e+02j,  2.02344422e+02-2.37198544e+02j,
    # -1.55053617e+02+4.95063597e+02j,  5.05625429e+02+7.17731979e+01j],
    # [ 7.23856406e+02-2.57973205e+02j, -1.97789098e+02-2.38177650e+02j,
    # 3.41994049e+02-1.43722635e-15j , 1.70595960e+02-1.40568891e+02j,
    # 2.65821125e+02-5.12284849e+02j, -3.42393449e+02-5.21353618e+00j,
    # 5.27852132e+02-2.16132116e+02j, -2.94246784e+02-4.78540067e+02j],
    # [ 4.69720262e+02+1.69492130e+02j, -6.17562632e-01-2.01077410e+02j,
    # 1.70595960e+02+1.40568891e+02j,  1.45442078e+02+4.30131025e-16j,
    # 3.44958135e+02-1.47139402e+02j, -1.69775268e+02-1.43926155e+02j,
    # 3.54256587e+02+1.09678576e+02j , 5.00216316e+01-3.61449790e+02j],
    # [ 9.53657842e+02+8.88213300e+02j,  2.04433876e+02-4.83299448e+02j,
    # 2.65821125e+02+5.12284849e+02j,  3.44958135e+02+1.47139402e+02j,
    # 9.80715260e+02+3.21936783e-15j, -2.59569616e+02-5.19315107e+02j,
    # 7.37422524e+02+6.25995605e+02j,  4.90812015e+02-8.16448144e+02j],
    # [-7.24412394e+02+2.70281966e+02j,  2.02344422e+02+2.37198544e+02j,
    # -3.42393449e+02+5.21353618e+00j, -1.69775268e+02+1.43926155e+02j,
    # -2.59569616e+02+5.19315107e+02j,  3.46570425e+02+3.98591700e-16j,
    # -5.28066855e+02+2.25339224e+02j,  3.03257510e+02+4.76937210e+02j],
    # [ 1.28687308e+03+5.90436586e+01j, -1.55053617e+02-4.95063597e+02j,
    # 5.27852132e+02+2.16132116e+02j,  3.54256587e+02-1.09678576e+02j,
    # 7.37422524e+02-6.25995605e+02j, -5.28066855e+02-2.25339224e+02j,
    # 9.58122381e+02+0.00000000e+00j, -1.52615050e+02-9.29074324e+02j],
    # [-2.62877100e+02+1.24084339e+03j,  5.05625429e+02-7.17731979e+01j,
    # -2.94246784e+02+4.78540067e+02j,  5.00216316e+01+3.61449790e+02j,
    # 4.90812015e+02+8.16448144e+02j,  3.03257510e+02-4.76937210e+02j,
    # -1.52615050e+02+9.29074324e+02j,  9.29159328e+02+0.00000000e+00j]]

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
    # RootMUSIC_Rx_eig = np.sort(np.real(LA.eigvals(RootMUSIC_Rx)))[::-1]
    # Deep_RootMUSIC_Rx_eig = np.sort(np.real(LA.eigvals(Deep_RootMUSIC_Rx))[0])[::-1]
    # algorithm = "ssn"
    # plt.style.use('default')
    # fig = plt.figure(figsize=(7, 5.5))
    # plt.style.use('plot_style.txt')

    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalues [λ]")
    # plt.xlim([0.85, 8.15])
    # plt.ylim([-0.02, 1.02])
    # plt.stem([i + 1 + 0.05 for i in range(RootMUSIC_Rx_eig.shape[0])],RootMUSIC_Rx_eig / np.max(RootMUSIC_Rx_eig), '#842ab0', label="R-MUSIC")
    # markerline, stemlines, baseline = plt.stem([i + 1 - 0.05 for i in range(Deep_RootMUSIC_Rx_eig.shape[0])],Deep_RootMUSIC_Rx_eig / np.max(Deep_RootMUSIC_Rx_eig),'#039403', markerfmt='>', label="SubNet+R-MUSIC")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # # plt.setp(stemlines, 'linestyle', 'dashed')
    # plt.legend()
    # plt.savefig("{}.pdf".format(simulation),bbox_inches='tight')

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
    