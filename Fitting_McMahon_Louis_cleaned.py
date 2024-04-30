
"""------------------
 Modules Importation
------------------"""
from scipy.stats import norm

print("Code started")


from symfit import *
from symfit.core.objectives import LogLikelihood
from sympy import posify, exp
from sympy import symbols
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from _collections import OrderedDict


"""-----------------------------------------------""" 


"""--------------------------------------------------
Configuration of LaTeX Text Rendering in Matplotlib
--------------------------------------------------"""
fig_width_pt    = 800.0                        # Get this from LaTeX using \showthe\columnwidth
inches_per_pt   = 1.0/72.27                     # Convert pt to inch
golden_mean     = (np.sqrt(5)-1.0)/2.0          # Aesthetic ratio
fig_width       = fig_width_pt*inches_per_pt    # Width in inches
fig_height      = fig_width*golden_mean         # Height in inches
fig_size        = [fig_width,fig_height]
params          = {'backend': 'ps',
                   'axes.labelsize': 20,
                   'font.size': 20,
                   'legend.fontsize': 20,
                   'xtick.labelsize': 18,
                   'ytick.labelsize': 18,
                   'font.family': 'Arial',
                   'figure.figsize': fig_size}
plt.rcParams.update(params)
"""------------------------------------------------------------------------------------------"""
def chi2_one_form(data, sd, model):
    return np.sum(((data-model)/sd)**2)

def chi2_all_forms(dataS, dataC, dataL, sdS, sdC, sdL, modelS, modelC, modelL):
    return 0

def z_test(data1, data2, sd1, sd2):
    return (data1-data2)/np.sqrt(sd1**2+sd2**2)

def z_test_significance(data1, data2, sd1, sd2):
    return 1-norm.cdf(z_test(data1, data2, sd1, sd2))

def n_stars(p_value):
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

#condition # 21% #1% #DMSO
#beam # Xray #Gantry1 #CLEAR #eRT6
def MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho(condition, beam, path_to_folder, delimiter = ";" ):

    supercoiled = True

    if condition == '21%':
        max_dose = 12

    elif condition == 'DMSO':
        max_dose = 55

    else:
        max_dose = 35

    file_name_UHDR = "UHDR-" + condition + "-" + beam + ".csv"
    file_name_CONV = "CONV-" + condition + "-" + beam + ".csv"

    if beam == "Xray":
        file_name_UHDR = file_name_CONV

    print(file_name_CONV)
    print(file_name_UHDR)

    file_path_CONV = path_to_folder + file_name_CONV
    file_path_UHDR = path_to_folder + file_name_UHDR

    print(file_path_CONV)
    print(file_path_UHDR)

    FLASH_data = pd.read_csv(file_path_UHDR, header=None, delimiter=delimiter)
    CONV_data = pd.read_csv(file_path_CONV, header=None, delimiter=delimiter)

    # Define symbolic variables
    d, S, C, L = symbols('d S C L')

    # Convert data to SymPy expressions
    CONV_dose = CONV_data.to_numpy()[:, 0]
    print('DEBUG', CONV_dose, max_dose)
    CONV_indexes = np.where(CONV_dose < max_dose)[0]
    CONV_dose = CONV_dose[CONV_indexes]
    print(CONV_indexes)
    CONV_fracS = CONV_data.to_numpy()[:, 1][CONV_indexes]
    CONV_errS = CONV_data.to_numpy()[:, 2][CONV_indexes]
    CONV_fracL = CONV_data.to_numpy()[:, 4][CONV_indexes]
    CONV_errL = CONV_data.to_numpy()[:, 5][CONV_indexes]
    CONV_fracC = CONV_data.to_numpy()[:, 7][CONV_indexes]
    CONV_errC = CONV_data.to_numpy()[:, 8][CONV_indexes]


    FLASH_dose = FLASH_data.to_numpy()[:, 0]
    FLASH_indexes = np.where(FLASH_dose < max_dose)[0]  # Assuming 1D array
    FLASH_dose = FLASH_dose[FLASH_indexes]
    FLASH_fracS = FLASH_data.to_numpy()[:, 1][FLASH_indexes]
    FLASH_errS = FLASH_data.to_numpy()[:, 2][FLASH_indexes]
    FLASH_fracL = FLASH_data.to_numpy()[:, 4][FLASH_indexes]
    FLASH_errL = FLASH_data.to_numpy()[:, 5][FLASH_indexes]
    FLASH_fracC = FLASH_data.to_numpy()[:, 7][FLASH_indexes]
    FLASH_errC = FLASH_data.to_numpy()[:, 8][FLASH_indexes]

    print("Choosen fitting method is FixedS0C0rho")

    #compare using Tukey's HCD the FLASH and CONV data for the same condition

    # Load datasets
    # FLASH_data = pd.read_csv (r"C:\Users\louis\OneDrive\Desktop\Doc Admin\CHUV-PSI\Plasmid_Fitting\Plasmid_Fitting\Pull-eRT6-upto2022.10.08\Pull_eRT6_1%_CONV.csv", delimiter=";", header=None)
    # CONV_data = pd.read_csv (r"C:\Users\louis\OneDrive\Desktop\Doc Admin\CHUV-PSI\Plasmid_Fitting\Plasmid_Fitting\Pull-eRT6-upto2022.10.08\Pull_eRT6_1%_UHDR.csv", delimiter=";", header=None)

    C0_CONV = CONV_fracC[0]
    C0_FLASH = FLASH_fracC[0]
    S0_CONV = CONV_fracS[0]
    S0_FLASH = FLASH_fracS[0]
    rho = 10/4361

    # Define the parameters from the variables
    b_S, b_D = parameters('b_S, b_D')
    d, S, C, L = variables('d, S, C, L')

    # Datasets structuration
    dataC = {S: CONV_fracS, C: CONV_fracC, L: CONV_fracL}
    dataC = OrderedDict(dataC.items())
    errC = {S: CONV_errS, C: CONV_errC, L: CONV_errL}
    errC = OrderedDict(errC.items())
    
    dataF = {S: FLASH_fracS, C: FLASH_fracC, L: FLASH_fracL}
    dataF = OrderedDict(dataF.items())
    errF = {S: FLASH_errS, C: FLASH_errC, L: FLASH_errL}
    errF = OrderedDict(errF.items())

    # Model definition
    model_dict_CONV = {
        S: S0_CONV*exp(-(b_S + b_D)*d),
        C: exp(-b_D*d)*(C0_CONV*exp(-0.5*b_S*b_S*rho*d*d) + S0_CONV*(exp(-0.5*b_S*b_S*rho*d*d)-exp(-b_S*d))),
        L: 1-(C0_CONV+S0_CONV)*exp(-(b_D*d + 0.5*b_S*b_S*rho*d*d))
    }

    model_dict_FLASH = {
        S: S0_FLASH * exp(-(b_S + b_D) * d),
        C: exp(-b_D * d) * (C0_FLASH * exp(-0.5 * b_S * b_S * rho * d * d) + S0_FLASH * (exp(-0.5 * b_S * b_S * rho * d * d) - exp(-b_S * d))),
        L: 1 - (C0_FLASH + S0_FLASH) * exp(-(b_D * d + 0.5 * b_S * b_S * rho * d * d))
    }

    def S_fct(d, b_S, b_D, S0, C0, rho):
        return S0 * np.exp(-(b_S + b_D) * d)

    def C_fct(d, b_S, b_D, S0, C0, rho):
        return np.exp(-b_D * d) * (C0 * np.exp(-0.5 * b_S * b_S * rho * d * d) + S0 * (np.exp(-0.5 * b_S * b_S * rho * d * d) - np.exp(-b_S * d)))

    def L_fct(d, b_S, b_D, S0, C0, rho):
        return 1 - (C0 + S0) * np.exp(-(b_D * d + 0.5 * b_S * b_S * rho * d * d))


    # Definition of limits for the parameters
    b_S.min, b_S.max = 0, 1
    b_D.min, b_D.max = 0, 1

    # Fitting the datasets
    CONV_fit = Fit(model_dict_CONV, d=CONV_dose, S=CONV_fracS, C=CONV_fracC, L=CONV_fracL)#, sigma_S = CONV_errS, sigma_C = CONV_errC, sigma_L=CONV_errL)
    CONV_fit_result = CONV_fit.execute()

    FLASH_fit = Fit(model_dict_FLASH, d=FLASH_dose, S=FLASH_fracS, C=FLASH_fracC, L=FLASH_fracL)#, sigma_S = FLASH_errS, sigma_C = FLASH_errC, sigma_L = FLASH_errL)
    FLASH_fit_result = FLASH_fit.execute()

    # Output file with fitting parameters
    #sys.stdout = open("CONV_fixedS0C0rho_fit_result.txt", "w")
    print("============== CONV =============================")
    print(CONV_fit_result)
    #sys.stdout = open("FLASH_fixedS0C0rho_fit_result.txt", "w")
    print("============== UHDR =============================")
    print(FLASH_fit_result)
    #sys.stdout.close()
    print("\n==================================================")

    # Plot data with fitting curves
    CONV_d_fit = np.linspace(0, max_dose, 10000)
    CONV_model_fit_max = CONV_fit.model(d=CONV_d_fit, **CONV_fit_result.params)._asdict()
    CONV_model_fit = CONV_fit.model(d=CONV_d_fit, **CONV_fit_result.params)._asdict()
    
    FLASH_d_fit = np.linspace(0, max_dose, 10000)
    FLASH_model_fit = FLASH_fit.model(d=FLASH_d_fit, **FLASH_fit_result.params)._asdict()

    lower_b_S_CONV = CONV_fit_result.value(b_S) - CONV_fit_result.stdev(b_S)
    upper_b_S_CONV = CONV_fit_result.value(b_S) + CONV_fit_result.stdev(b_S)
    lower_b_D_CONV = CONV_fit_result.value(b_D) - CONV_fit_result.stdev(b_D)
    upper_b_D_CONV = CONV_fit_result.value(b_D) + CONV_fit_result.stdev(b_D)
    lower_b_S_FLASH = FLASH_fit_result.value(b_S) - FLASH_fit_result.stdev(b_S)
    upper_b_S_FLASH = FLASH_fit_result.value(b_S) + FLASH_fit_result.stdev(b_S)
    lower_b_D_FLASH = FLASH_fit_result.value(b_D) - FLASH_fit_result.stdev(b_D)
    upper_b_D_FLASH = FLASH_fit_result.value(b_D) + FLASH_fit_result.stdev(b_D)

    print("CONV")
    print(lower_b_S_CONV, upper_b_S_CONV, lower_b_D_CONV, upper_b_D_CONV)
    print("FLASH")
    print(lower_b_S_FLASH, upper_b_S_FLASH, lower_b_D_FLASH, upper_b_D_FLASH)


    with open(condition + "-" + beam + '.txt', 'w') as file:
        file.write("============== CONV =============================")
        file.write(str(CONV_fit_result))
        #sys.stdout = open("FLASH_fixedS0C0rho_fit_result.txt", "w")
        file.write("\n ============== UHDR =============================")
        file.write(str(FLASH_fit_result))
        #sys.stdout.close()
        file.write("\n ==================================================")
        file.write("\n" + str(CONV_fit_result.value(b_D)))
        file.write("\n" +str(CONV_fit_result.stdev(b_D)))
        file.write("\n" +str(CONV_fit_result.value(b_S)))
        file.write("\n" +str(CONV_fit_result.stdev(b_S)))
        file.write("\n" +str(FLASH_fit_result.value(b_D)))
        file.write("\n" +str(FLASH_fit_result.stdev(b_D)))
        file.write("\n" +str(FLASH_fit_result.value(b_S)))
        file.write("\n" +str(FLASH_fit_result.stdev(b_S)))
        p_beta_S = z_test_significance(CONV_fit_result.value(b_S),FLASH_fit_result.value(b_S),
                          CONV_fit_result.stdev(b_S),FLASH_fit_result.stdev(b_S))
        p_beta_D = z_test_significance(CONV_fit_result.value(b_D),FLASH_fit_result.value(b_D),
                            CONV_fit_result.stdev(b_D),FLASH_fit_result.stdev(b_D))

        file.write("\n beta_S significance " +str(p_beta_S) + n_stars(p_beta_S))
        file.write("\n beta_D significance " +str(p_beta_D) + n_stars(p_beta_D))
    
    r_squared_CONV = CONV_fit_result.r_squared
    r_squared_FLASH = FLASH_fit_result.r_squared
    
    if (r_squared_CONV < 0.95 or r_squared_FLASH < 0.95):
        print(" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n FITTING WAS NOT ACCURATE \n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    
    plt.figure()
    ax = plt.axes()
    Label = ['Supercoiled','Open Circular','Linear']
    
    Linestyle = ['-','--',':']
    marker_shape = ['o','s','^']

    # condition # 21% #1% #DMSO
    # beam # Xray #Gantry1 #CLEAR #eRT6

    #check if beam contains one of the following strings
    if 'Xray' in beam:
        color_CONV = (0, 0, 0, 1.0)  # RGBA color
        color_UHDR = (0, 0, 0, 1.0)  # RGBA color
    elif 'PSI' in beam:
        #color_CONV = 'darkorange'
        #color_UHDR = 'brown'
            color_CONV = (0 / 255, 192 / 255, 0 / 255, 1.0)  # RGBA color
            color_UHDR = (173 / 255, 8 / 255, 226 / 255, 0.996)  # RGBA color
    elif 'CLEAR' in beam:
        #color_CONV = 'cornflowerblue'
        #color_UHDR = 'navy'
        color_CONV = (90 / 255, 155 / 255, 234 / 255, 0.843)  # RGBA color
        color_UHDR = (237 / 255, 100 / 255, 17 / 255, 0.81)  # RGBA color
    elif 'eRT6' in beam:
        #color_CONV = 'limegreen'
        #color_UHDR = 'darkgreen'
        color_CONV = (1 / 255, 1 / 255, 253 / 255, 0.984)  # RGBA color
        color_UHDR = (255 / 255, 0 / 255, 0 / 255, 1.0)  # RGBA color

    # Plotting the data
    #set figure size



    i = 0
    for var in dataC:
        #plt.plot(CONV_dose,dataC[var],Marker[i],markersize=8,label=Label[i]+":    CONV")
        plt.errorbar(CONV_dose,dataC[var],marker= marker_shape[i], linestyle = '', color = color_CONV, markersize=3,mew=3,yerr=errC[var],ecolor=color_CONV,capsize=3)#,label=Label[i]+":    CONV")
        #plt.errorbar(CONV_dose[0],dataC[var][0], markersize=8,mew=3,yerr=errC[var],ecolor=c,capsize=3)#,label=Label[i]+":    CONV")
        plt.plot(CONV_d_fit,CONV_model_fit[var],color= color_CONV, linestyle = Linestyle[i],linewidth=2)
        i+=1
        
    j = 0
    for var in dataF:
        if not beam == 'Xray':
            #plt.plot(FLASH_dose,dataF[var],Marker[j],alpha=0.3,markersize=8,label=Label[j]+":    FLASH")
            plt.errorbar(FLASH_dose,dataF[var],marker= marker_shape[j], linestyle = '', color = color_UHDR, markersize=2,mew=3,yerr=errF[var],ecolor=color_UHDR,capsize=3)#,label=Label[i]+":    CONV")
            #plt.errorbar(FLASH_dose[0],dataF[var][0], markersize=8,mew=3,yerr=errF[var],ecolor='r',capsize=3)#,label=Label[j]+":    UHDR")
            plt.plot(FLASH_d_fit,FLASH_model_fit[var],color= color_UHDR,linestyle = Linestyle[j],linewidth=2)
            j+=1

    # plt.fill_between(CONV_d_fit, L_fct(CONV_d_fit, lower_b_S_CONV, lower_b_D_CONV, S0_CONV, C0_CONV, rho),
    #                  L_fct(CONV_d_fit, upper_b_S_CONV, upper_b_D_CONV, S0_CONV, C0_CONV, rho), color=color_CONV,
    #                  alpha=0.3)
    # plt.fill_between(FLASH_d_fit, L_fct(FLASH_d_fit, lower_b_S_FLASH, lower_b_D_FLASH, S0_FLASH, C0_FLASH, rho),
    #                  L_fct(FLASH_d_fit, upper_b_S_FLASH, upper_b_D_FLASH, S0_FLASH, C0_FLASH, rho), color=color_UHDR,
    #                  alpha=0.3)
    # plt.fill_between(CONV_d_fit, C_fct(CONV_d_fit, lower_b_S_CONV, lower_b_D_CONV, S0_CONV, C0_CONV, rho),
    #                  C_fct(CONV_d_fit, upper_b_S_CONV, upper_b_D_CONV, S0_CONV, C0_CONV, rho), color=color_CONV,
    #                  alpha=0.3)
    # plt.fill_between(FLASH_d_fit, C_fct(FLASH_d_fit, lower_b_S_FLASH, lower_b_D_FLASH, S0_FLASH, C0_FLASH, rho),
    #                  C_fct(FLASH_d_fit, upper_b_S_FLASH, upper_b_D_FLASH, S0_FLASH, C0_FLASH, rho), color=color_UHDR,
    #                  alpha=0.3)
    # plt.fill_between(CONV_d_fit, S_fct(CONV_d_fit, lower_b_S_CONV, lower_b_D_CONV, S0_CONV, C0_CONV, rho),
    #                  S_fct(CONV_d_fit, upper_b_S_CONV, upper_b_D_CONV, S0_CONV, C0_CONV, rho), color=color_CONV,
    #                  alpha=0.3)
    # plt.fill_between(FLASH_d_fit, S_fct(FLASH_d_fit, lower_b_S_FLASH, lower_b_D_FLASH, S0_FLASH, C0_FLASH, rho),
    #                  S_fct(FLASH_d_fit, upper_b_S_FLASH, upper_b_D_FLASH, S0_FLASH, C0_FLASH, rho), color=color_UHDR,
    #                  alpha=0.3)

    ax.set_xlim([0,max_dose])
    ax.set_ylim([0,1])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.grid()
    ax.patch.set_color("w")
    ax.set_xlabel("Dose  [Gy] ")
    ax.set_ylabel("Fraction")
    plt.savefig("Plots/plot_fixed_" + condition + "-" + beam + ".tiff", dpi=600)
    plt.show()

    print("Figure printed and saved \n ---------------------------------------------------------------------")


    summary_doses = np.linspace(3,5, 21)
    for dose in summary_doses:
        print("Dose: ", dose)
        c = C_fct(dose, CONV_fit_result.value(b_S), CONV_fit_result.value(b_D), S0_CONV, C0_CONV, rho)
        f = C_fct(dose, FLASH_fit_result.value(b_S), FLASH_fit_result.value(b_D), S0_FLASH, C0_FLASH, rho)
        print("CONV: ", c)
        print("FLASH: ", f)
        print("mean: ", (c+f)/2)


path = 'mean&SD/'
#
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI",path, delimiter = ',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","CLEAR",path, delimiter = ',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","eRT6",path)
# #
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%b","PSI",path)
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","CLEAR",path, delimiter = ',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","eRT6",path, delimiter = ',')
# #
# # MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","PSI",path)
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","CLEAR",path, delimiter = ',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","eRT6",path)
# #
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("0uM","eRT6",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1,5uM","eRT6",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("5uM","eRT6",path,delimiter=',')
# #
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe0uM","PSI",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe1.5uM","PSI",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe5uM","PSI",path,delimiter=',')
# #
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph4","PSI",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph5","PSI",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph7","PSI",path)
# #
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI-SOBP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI-BP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","PSI-BP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","PSI-SOBP",path,delimiter=',')

# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSOx10","Xray",path)

#3MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","Xray",path, delimiter=';')

#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("0uM","Xray",path, delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1,5uM","Xray",path, delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("5uM","Xray",path, delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("10uM","Xray",path, delimiter=',')

