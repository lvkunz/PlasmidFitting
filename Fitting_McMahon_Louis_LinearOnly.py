
"""------------------
 Modules Importation
------------------"""

print("Code started")


from symfit import *
from sympy import posify, exp
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from _collections import OrderedDict
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, chi2, norm

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
params          = {'axes.labelsize': 20,
                   'font.size': 20,
                   'legend.fontsize': 20,
                   'xtick.labelsize': 18,
                   'ytick.labelsize': 18,
                   'font.family': 'Arial',
                   'figure.figsize': fig_size}
plt.rcParams.update(params)
"""------------------------------------------------------------------------------------------"""


def chi2_one_form(data, sd, n, model):
    chi_squared = 0

    for i in range(len(data)):
        if sd[i] != 0:  # Additional check to avoid division by zero
            degrees_of_freedom_i = n[i] - 1  # degrees of freedom for each term
            #chi_squared += n[i]*(((data[i] - model[i]) / sd[i]) ** 2)
            chi_squared += n[i]*((data[i] - model[i])) ** 2


    return chi_squared

def chi2_all_forms(dataS, dataC, dataL, sdS, sdC, sdL, n, modelS, modelC, modelL):
    proportions = [2/3, 2/3, 2/3]
    return proportions[0]*chi2_one_form(dataS, sdS, n, modelS) + proportions[1]*chi2_one_form(dataC, sdC, n, modelC) + proportions[2]*chi2_one_form(dataL, sdL, n, modelL)


def model_L(d, S0, C0, b_D):
    L0 = 1 - S0 - C0
    L = L0 + b_D * d
    return L

def objective(params, dataS, dataC, dataL, sdS, sdC, sdL, n, S0, C0, rho, d):
    # Extract parameters
    b_D = params

    # Extract model predictions
    modelL = model_L(d, S0, C0, b_D)

    # Calculate the chi-squared value
    chi_squared = chi2_one_form(dataL,sdL,n,modelL)

    return chi_squared

def calculate_negative_log_likelihood(params, dataS, dataC, dataL, sdS, sdC, sdL, n, S0, C0, rho, d):

    b_D = params

    modelL = model_L(d, S0, C0, b_D)

    log_likelihood_L = np.log(norm.pdf(dataL, loc=modelL, scale=sdL))

    # Replace nan and inf values with 0
    log_likelihood_L = np.nan_to_num(log_likelihood_L, nan=0, posinf=1e3, neginf=-1e3)

    return -np.sum(log_likelihood_L)

def n_stars(ttest):
    n_stars = int(abs(ttest)-2.0)
    return n_stars

def read_csv_file(file_path, delimiter=','):
    try:
        df = pd.read_csv(file_path, header=None, delimiter=delimiter)
        # Do something with the DataFrame 'df'
        print(f"Successfully read {file_path}")
        return df
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty or does not contain any valid data.")
        return None
    except FileNotFoundError:
        print(f"File {file_path} not found. Check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

def resample_data(fracS, fracC, fracL, sdS, sdC, sdL, n, S0, C0, rho, d):
    # Assuming you want to resample the data by adding some random noise
     resampled_dataS = fracS + np.random.normal(0, 2 * sdS, len(fracS))
     resampled_dataC = fracC + np.random.normal(0, 2 * sdC, len(fracC))
     resampled_dataL = fracL + np.random.normal(0, 2 * sdL, len(fracL))

     return resampled_dataS, resampled_dataC, resampled_dataL, sdS, sdC, sdL, n, S0, C0, rho, d

# Continue with further processing or analysis based on FLASH_data and CONV_data

#condition # 21% #1% #DMSO
#beam # Xray #Gantry1 #CLEAR #eRT6
def MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho(condition, beam, path_to_folder, delimiter = ";" ):

    supercoiled = True

    if condition == '21%':
        max_dose = 12
    else:
        max_dose = 55

    file_name_UHDR = "UHDR-" + condition + "-" + beam + ".csv"
    file_name_CONV = "CONV-" + condition + "-" + beam + ".csv"

    if beam == "Xray":
        file_name_UHDR = file_name_CONV

    print(file_name_CONV)
    print(file_name_UHDR)

    file_path_CONV = path_to_folder + file_name_CONV
    file_path_UHDR = path_to_folder + file_name_UHDR

    # Try reading with both delimiters

    delimiter = ';'

    FLASH_data = read_csv_file(file_path_UHDR, delimiter)
    CONV_data = read_csv_file(file_path_CONV, delimiter)

    if not FLASH_data.shape[1] > 1:
        FLASH_data = read_csv_file(file_path_UHDR, ',')
        CONV_data = read_csv_file(file_path_CONV, ',')

    #normal
    CONV_dose = np.array(CONV_data.to_numpy()[:, 0])
    CONV_fracS = np.array(CONV_data.to_numpy()[:, 1])
    CONV_errS = np.array(CONV_data.to_numpy()[:, 2])
    CONV_fracL = np.array(CONV_data.to_numpy()[:, 4])
    CONV_errL = np.array(CONV_data.to_numpy()[:, 5])
    CONV_fracC = np.array(CONV_data.to_numpy()[:, 7])
    CONV_errC = np.array(CONV_data.to_numpy()[:, 8])

    FLASH_dose = np.array(FLASH_data.to_numpy()[:, 0])
    FLASH_fracS = np.array(FLASH_data.to_numpy()[:, 1])
    FLASH_errS = np.array(FLASH_data.to_numpy()[:, 2])
    FLASH_fracL = np.array(FLASH_data.to_numpy()[:, 4])
    FLASH_errL = np.array(FLASH_data.to_numpy()[:, 5])
    FLASH_fracC = np.array(FLASH_data.to_numpy()[:, 7])
    FLASH_errC = np.array(FLASH_data.to_numpy()[:, 8])

    nCONV = np.array(CONV_data.to_numpy()[:, 3])
    nFLASH = np.array(FLASH_data.to_numpy()[:, 3])

    print("Choosen fitting method is FixedS0C0rho - personallized chi 2")

    C0_CONV = CONV_fracC[0]
    C0_FLASH = FLASH_fracC[0]
    S0_CONV = CONV_fracS[0]
    S0_FLASH = FLASH_fracS[0]
    rho = 10/4361

    initial_params = [0.005]
    if condition == '21%':
        initial_params = [0.005]
    elif condition == '1%':
        initial_params = [0.003]
    elif condition == 'DMSO':
        initial_params = [0.0003]

    bounds = [(0.0, 0.1)]
    #bounds = [(initial_params[0]*0.1, initial_params[0]*10), (initial_params[1]*0.1, initial_params[1]*10)]
    # Function to perform bootstrap iterations

    method = 'BFGS'
    # Datasets structuration
    result_CONV = minimize(
        objective,
        initial_params,
        bounds = bounds,
        args=(CONV_fracS, CONV_fracC, CONV_fracL, CONV_errS, CONV_errC, CONV_errL, nCONV, S0_CONV, C0_CONV, rho, CONV_dose),
        method = method
    )

    result_FLASH = minimize(
        objective,
        initial_params,
        bounds = bounds,
        args=(FLASH_fracS, FLASH_fracC, FLASH_fracL, FLASH_errS, FLASH_errC, FLASH_errL, nFLASH, S0_FLASH, C0_FLASH, rho, FLASH_dose),
        method = method
    )

    # Extract the optimized parameters
    optimized_params_CONV = result_CONV.x
    optimized_params_FLASH = result_FLASH.x

    #compute the Delta chi 2
    Delta_chi2 = objective(optimized_params_CONV, CONV_fracS, CONV_fracC, CONV_fracL, CONV_errS, CONV_errC, CONV_errL, nCONV, S0_CONV, C0_CONV, rho, CONV_dose)\
                 - objective(optimized_params_FLASH, FLASH_fracS, FLASH_fracC, FLASH_fracL, FLASH_errS, FLASH_errC, FLASH_errL, nFLASH, S0_FLASH, C0_FLASH, rho, FLASH_dose)
    print("Delta chi 2: ", Delta_chi2)

    df = 2  # Replace with the appropriate degrees of freedom
    p_value = 1 - chi2.cdf(Delta_chi2, df)

    print("p-value chi2: ", p_value)

    print("Optimized parameters for CONV: ", optimized_params_CONV)
    print("Optimized parameters for FLASH: ", optimized_params_FLASH)

    #Compute the Hessian matrix
    hessian_CONV = result_CONV.hess_inv
    covariance_matrix_CONV = hessian_CONV * 2.0
    parameter_standard_errors_CONV = np.sqrt(np.diag(covariance_matrix_CONV))

    hessian_FLASH = result_FLASH.hess_inv
    covariance_matrix_FLASH = hessian_FLASH * 2.0
    parameter_standard_errors_FLASH = np.sqrt(np.diag(covariance_matrix_FLASH))

    #parameter_standard_errors_CONV = [1,1]
    #parameter_standard_errors_FLASH = [1,1]

    print(condition + beam)
    print("DSB CONV:", optimized_params_CONV[0], '+-', parameter_standard_errors_CONV[0])
    #print("DSB CONV:", optimized_params_CONV[1], '+-', parameter_standard_errors_CONV[1])
    #print("S0 CONV:", optimized_params_CONV[2], '+-', parameter_standard_errors_CONV[2])
    #print("C0 CONV:", optimized_params_CONV[3], '+-', parameter_standard_errors_CONV[3])

    # Extract the optimized parameters
    print("DSB FLASH:", optimized_params_FLASH[0], '+-', parameter_standard_errors_FLASH[0])
    #print("DSB FLASH:", optimized_params_FLASH[1], '+-', parameter_standard_errors_FLASH[1])
    #print("S0 FLASH:", optimized_params_FLASH[2], '+-', parameter_standard_errors_FLASH[2])
    #print("C0 FLASH:", optimized_params_FLASH[3], '+-', parameter_standard_errors_FLASH[3])

    # # Output file with fitting parameters
    with open(condition + "-" + beam + '.txt', 'w') as file:
        file.write("============== CONV =============================")
        file.write("\nDSB CONV:" + str(optimized_params_CONV[0]) + '+-' + str(parameter_standard_errors_CONV[0]))
        #file.write("\nDSB CONV:" + str(optimized_params_CONV[1]) + '+-' + str(parameter_standard_errors_CONV[1]))
        #file.write("\nS0 CONV:" + str(optimized_params_CONV[2]) + '+-' + str(parameter_standard_errors_CONV[2]))
        #file.write("\nC0 CONV:" + str(optimized_params_CONV[3]) + '+-' + str(parameter_standard_errors_CONV[3]))
        #sys.stdout = open("FLASH_fixedS0C0rho_fit_result.txt", "w")
        file.write("\n ============== UHDR =============================")
        file.write("\nDSB FLASH:" + str(optimized_params_FLASH[0]) + '+-' + str(parameter_standard_errors_FLASH[0]))
        #.write("\nDSB FLASH:" + str(optimized_params_FLASH[1]) + '+-' + str(parameter_standard_errors_FLASH[1]))
        #file.write("\nS0 FLASH:" + str(optimized_params_FLASH[2]) + '+-' + str(parameter_standard_errors_FLASH[2]))
        #file.write("\nC0 FLASH:" + str(optimized_params_FLASH[3]) + '+-' + str(parameter_standard_errors_FLASH[3]))
        file.write("\n ==================================================")
        file.write("\n")
        file.close()


    # Plot data with fitting curves
    CONV_d_fit = np.linspace(0, max_dose, 10000)
    FLASH_d_fit = np.linspace(0, max_dose, 10000)

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


    # #plot supercoiled
    # plt.errorbar(CONV_dose, CONV_fracS, marker=marker_shape[0], linestyle='', color=color_CONV, markersize=3,
    #                 mew=3,yerr=CONV_errS, ecolor=color_CONV, capsize=3)
    # plt.plot(CONV_d_fit, model_S(CONV_d_fit, S0, C0 , rho, optimized_params_CONV[0], optimized_params_CONV[1]),
    #              color=color_CONV, linestyle=Linestyle[0], linewidth=2)
    # plt.errorbar(FLASH_dose, FLASH_fracS, marker=marker_shape[0], linestyle='', color=color_UHDR, markersize=3,
    #              mew=3, yerr=FLASH_errS, ecolor=color_UHDR, capsize=3)
    # plt.plot(FLASH_d_fit, model_S(FLASH_d_fit, S0, C0 , rho, optimized_params_FLASH[0], optimized_params_FLASH[1]),
    #          color=color_UHDR, linestyle=Linestyle[0], linewidth=2)

    # #plot open circular
    # plt.errorbar(CONV_dose, CONV_fracC, marker=marker_shape[1], linestyle='', color=color_CONV, markersize=3,
    #                 mew=3,yerr=CONV_errC, ecolor=color_CONV, capsize=3)
    # plt.plot(CONV_d_fit, model_C(CONV_d_fit, S0_CONV, C0_CONV , rho, optimized_params_CONV[0], optimized_params_CONV[1]),
    #                 color=color_CONV, linestyle=Linestyle[1], linewidth=2)
    # plt.errorbar(FLASH_dose, FLASH_fracC, marker=marker_shape[1], linestyle='', color=color_UHDR, markersize=3,
    #                 mew=3, yerr=FLASH_errC, ecolor=color_UHDR, capsize=3)
    # plt.plot(FLASH_d_fit, model_C(FLASH_d_fit, S0_FLASH, C0_FLASH , rho, optimized_params_FLASH[0], optimized_params_FLASH[1]),
    #                 color=color_UHDR, linestyle=Linestyle[1], linewidth=2)

    #plot linear
    plt.errorbar(CONV_dose, CONV_fracL, marker=marker_shape[2], linestyle='', color=color_CONV, markersize=3,
                    mew=3,yerr=CONV_errL, ecolor=color_CONV, capsize=3)
    plt.plot(CONV_d_fit, model_L(CONV_d_fit, S0_CONV, C0_CONV , optimized_params_CONV[0]),
                    color=color_CONV, linestyle=Linestyle[2], linewidth=2)
    plt.errorbar(FLASH_dose, FLASH_fracL, marker=marker_shape[2], linestyle='', color=color_UHDR, markersize=3,
                    mew=3, yerr=FLASH_errL, ecolor=color_UHDR, capsize=3)
    plt.plot(FLASH_d_fit, model_L(FLASH_d_fit, S0_FLASH, C0_FLASH ,optimized_params_FLASH[0]),
                    color=color_UHDR, linestyle=Linestyle[2], linewidth=2)

    plt.title(str(condition))
    ax.set_xlim([-0.1,max_dose])
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.grid()
    ax.patch.set_color("w")
    ax.set_xlabel("Dose  [Gy] ")
    ax.set_ylabel("Fraction")
    plt.savefig('Plots/improvedMethod/' + str(condition) + str(beam) + '.png', dpi=400, bbox_inches='tight')
    plt.show()

    # with(open('Plots/improvedMethod/' + condition + "-" + beam + '.txt', 'w')) as file:
    #     file.write('SSB-CONV, DSB-CONV, sdSSB-CONV, sdDSB-CONV, SSB-FLASH, DSB-FLASH, sdSSB-FLASH, sdDSB-FLASH, ttest_SSB, ttest_DSB \n')
    #     file.write(str(optimized_params_CONV[0]) + "\n")
    #     file.write(str(optimized_params_CONV[1]) + "\n")
    #     file.write(str(parameter_standard_errors_CONV[0]) + "\n")
    #     file.write(str(parameter_standard_errors_CONV[1]) + "\n")
    #     file.write(str(optimized_params_FLASH[0]) + "\n")
    #     file.write(str(optimized_params_FLASH[1]) + "\n")
    #     file.write(str(parameter_standard_errors_FLASH[0]) + "\n")
    #     file.write(str(parameter_standard_errors_FLASH[1]) + "\n")
    #     file.write(str(ttest_SSB) + '*'* n_stars(ttest_SSB)+ "\n")
    #     file.write(str(ttest_DSB) + '*'*n_stars(ttest_DSB)+ "\n")
    #     file.close()

    print("Figure printed and saved \n ---------------------------------------------------------------------")


path = 'mean&SD/'

MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe0uM","PSI",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe1.5uM","PSI",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("Fe5uM","PSI",path,delimiter=',')
#
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","CLEAR",path)
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","eRT6",path)

#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","PSI",path,delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","CLEAR",path)
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","eRT6",path)

# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","Xray",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","PSI",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","CLEAR",path)
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","eRT6",path)
# #
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("0uM","eRT6",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1,5uM","eRT6",path,delimiter=',')
# MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("5uM","eRT6",path,delimiter=',')
#
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("0uM","PSI",path,delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1,5uM","PSI",path,delimiter=',')
#MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("3uM","PSI",path,delimiter=',')
#
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph4","PSI",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph5","PSI",path)
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("ph7","PSI",path)
#
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI-SOBP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("21%","PSI-BP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("DMSO","PSI-BP",path,delimiter=',')
MAGIC_Plasmid_CurveFitting_NCh_FixedS0C0rho("1%","PSI-SOBP",path,delimiter=',')
#





