import math
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set the path to the folder containing your files
directory_path = r"C:\Users\louis\PycharmProjects\pythonProject1"

# Initialize an empty DataFrame to store the data
df = pd.DataFrame()

def z_test(data1, data2, sd1, sd2):
    return abs((data1-data2)/np.sqrt(sd1**2+sd2**2))

def z_test_significance(data1, data2, sd1, sd2):
    return 1-norm.cdf(z_test(data1, data2, sd1, sd2))

def n_stars(p_value):
    if p_value < 0.0001:
        return 4
    elif p_value < 0.001:
        return 3
    elif p_value < 0.01:
        return 2
    elif p_value < 0.05:
        return 1
    else:
        return 0


keywords = ['1%b-PSI','1%-PSI-SOBP','21%-PSI','21%-PSI-SOBP','21%-PSI-BP','DMSO-PSI','DMSO-PSI-BP']  # Add the keywords you want to filter by
filtered_file_paths = []


for filename in os.listdir(directory_path):
    # Check if any of the keywords are present in the filename
    if any(keyword in filename for keyword in keywords):
        if filename.endswith('.txt'):  # Adjust the file extension as needed
            # If yes, construct the full file path and add it to the list
            file_path = os.path.join(directory_path, filename)
            filtered_file_paths.append(file_path)

# Loop through each file in the folder
for filename in filtered_file_paths:
    with open(filename, 'r') as file:
        lines = file.readlines()
        parameter_name = filename
        #get the name of the .txt file
        condition = filename.split("\\")[-1].split("_")[0]
        #get the condition from the name of the .txt file
        condition = condition.split(".")[0]
        DSB_CONV = float(lines[-10].strip())  # Assuming a single value per file
        sdDSB_CONV = float(lines[-9].strip())  # Assuming a single value per file
        SSB_CONV = float(lines[-8].strip())  # Assuming a single value per file
        sdSSB_CONV = float(lines[-7].strip())  # Assuming a single value per file
        DSB_FLASH = float(lines[-6].strip())  # Assuming a single value per file
        sdDSB_FLASH = float(lines[-5].strip())  # Assuming a single value per file
        SSB_FLASH = float(lines[-4].strip())  # Assuming a single value per file
        sdSSB_FLASH = float(lines[-3].strip())  # Assuming a single value per file
        p_value_DSB = z_test_significance(DSB_CONV, DSB_FLASH, sdDSB_CONV, sdDSB_FLASH)
        p_value_SSB = z_test_significance(SSB_CONV, SSB_FLASH, sdSSB_CONV, sdSSB_FLASH)
        # Create a DataFrame with the extracted information
        temp_df = pd.DataFrame({
            'Condition': [condition],
            'SSB CONV': [SSB_CONV],
            'DSB CONV': [DSB_CONV],
            'sdSSB CONV': [sdSSB_CONV],
            'sdDSB CONV': [sdDSB_CONV],
            'SSB FLASH': [SSB_FLASH],
            'DSB FLASH': [DSB_FLASH],
            'sdSSB FLASH': [sdSSB_FLASH],
            'sdDSB FLASH': [sdDSB_FLASH],
            'p_value SSB': [p_value_SSB],
            'p_value DSB': [p_value_DSB],
            'n_stars SSB': [n_stars(p_value_SSB)],
            'n_stars DSB': [n_stars(p_value_DSB)],
            'ratio SSB': [(SSB_CONV - SSB_FLASH)],
            'ratio DSB': [(DSB_CONV - DSB_FLASH)],
            'ratio sdSSB': [np.sqrt(sdSSB_CONV**2+sdSSB_FLASH**2)],
            'ratio sdDSB': [np.sqrt(sdDSB_CONV**2+sdDSB_FLASH**2)],
            'fold change SSB': [math.log2(SSB_CONV/SSB_FLASH) if (SSB_CONV > 0 and SSB_FLASH > 0) else 0.0],
            'fold change DSB': [math.log2(DSB_CONV/DSB_FLASH) if (DSB_CONV > 0 and DSB_FLASH > 0) else 0.0],
            'fold change sdSSB': [0],
            'fold change sdDSB': [0]
        })
        # Append the temporary DataFrame to the main DataFrame
        df = pd.concat([df, temp_df], ignore_index=True)

# Save the DataFrame to a CSV file
name = ''
print(df)
for keyword in keywords:
    name += keyword + '_'

name += 'data.csv'

df.to_csv(name, sep=',', index=True)
capsize = 3

#df.plot(x='Condition', y=['ratio SSB', 'ratio DSB'], kind='bar', yerr=['ratio sdSSB', 'ratio sdDSB'], capsize=capsize)

# Assume df contains the data

# Create the plot
logP_DSB = -np.log10(df['p_value DSB'])
logP_SSB = -np.log10(df['p_value SSB'])

p_value_threshold = 1.3
logFold_threshold = 0.6

#line range for the p-value is total minimum to total maximum
top_stop = max(max(logP_DSB), max(logP_SSB), p_value_threshold)
bottom_stop = 0.0
left_stop = min(min(df['fold change DSB']), min(df['fold change SSB']),-logFold_threshold)
right_stop = max(max(df['fold change DSB']), max(df['fold change SSB']),logFold_threshold)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.errorbar(df['fold change DSB'], range(len(df)), xerr=df['fold change sdDSB'], fmt='o', label='DSB', capsize=capsize, color='black')
ax1.set_yticks(range(len(df)))

#ax1.set_yticklabels(["1% Oxygen,\n Fe(II) 0uM","1% Oxygen,\n Fe(II) 1.5uM","1% Oxygen,\n Fe(II) 5uM"])
#ax1.set_yticklabels(["1% Oxygen,\n pH 4 ","1% Oxygen,\n pH 5 ","1% Oxygen,\n pH 7"])

ax1.vlines(0, -1, len(df), colors='grey', linestyles='dashed')
ax1.set_xlabel("$\\beta_{CDR} - \\beta_{UHDR}$", fontsize=15)
ax1.set_title('Difference of $\\beta_{DSB}$')

ax2.errorbar(df['fold change SSB'], range(len(df)), xerr=df['fold change sdSSB'], fmt='s', label='SSB', capsize=capsize, color='black')
ax2.set_yticks(range(len(df)))
#ax2.set_yticklabels(["","",""])
ax2.vlines(0, -1, len(df), colors='grey', linestyles='dashed')
ax2.set_xlabel("$\\beta_{CDR} - \\beta_{UHDR}$", fontsize=15)
ax2.set_title('Difference of $\\beta_{SSB}$')

plt.tight_layout()
plt.savefig('ratios_simple.png', dpi=600)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.errorbar( df['fold change DSB'],logP_DSB, fmt='o', label='DSB', capsize=capsize, color='black')
ax1.vlines(-logFold_threshold, bottom_stop, top_stop, colors='grey', linestyles='dashed')
ax1.vlines(logFold_threshold, bottom_stop, top_stop, colors='grey', linestyles='dashed')
ax1.hlines(p_value_threshold, left_stop, right_stop, colors='grey', linestyles='dashed')
ax1.set_xlabel("$\\log_2(\\beta_{CONV}/\\beta_{UHDR})$", fontsize=15)
ax1.set_ylabel("$-\\log_{10}(p_{value})$", fontsize=15)
ax1.set_title('$\\beta_{DSB}$')

for x, y, condition in zip(df['fold change DSB'], logP_DSB, df['Condition']):
    color = 'black'  # default color
    if '21%' in condition:
        color = 'blue'
    elif '1%' in condition:
        color = 'red'
    elif 'DMSO' in condition:
        color = 'green'
    elif 'uM' in condition:
        color = 'red'
    elif 'ph' in condition:
        color = 'red'

    ax1.text(x, y, condition, fontsize=8, ha='center', va='bottom', color=color)


ax2.errorbar( df['fold change SSB'],logP_SSB, fmt='s', label='SSB', capsize=capsize, color='black')
ax2.vlines(-logFold_threshold, bottom_stop, top_stop, colors='grey', linestyles='dashed')
ax2.vlines(logFold_threshold, bottom_stop, top_stop, colors='grey', linestyles='dashed')
ax2.hlines(p_value_threshold, left_stop, right_stop, colors='grey', linestyles='dashed')
ax2.set_xlabel("$\\log_2(\\beta_{CONV}/\\beta_{UHDR})$", fontsize=15)
ax2.set_ylabel("$-\\log_{10}(p_{value})$", fontsize=15)
ax2.set_title('$\\beta_{SSB}$')

for x, y, condition in zip(df['fold change SSB'], logP_SSB, df['Condition']):
    color = 'black'  # default color
    if '21%' in condition:
        color = 'blue'
    elif '1%' in condition:
        color = 'red'
    elif 'DMSO' in condition:
        color = 'green'
    elif 'uM' in condition:
        color = 'red'
    elif 'ph' in condition:
        color = 'red'

    ax2.text(x, y, condition, fontsize=8, ha='center', va='bottom', color=color)

plt.tight_layout()
plt.savefig('volcano.png', dpi=600)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 5))

color_CONV = (0 / 255, 192 / 255, 0 / 255, 1.0)  # RGBA color
color_UHDR = (173 / 255, 8 / 255, 226 / 255, 0.996)  # RGBA color

pH_array = [4.0, 5.0, 7.0]
Fe_array = [0.0, 1.5, 5.0]
labels = df['Condition']
labels = ['1% O2,\n SOBP', '1% O2,\n TR', '21% O2,\n BP', '21% O2,\n SOBP', '21% O2,\n TR', '21% O2,\n DMSO 14mM BP', '21% O2,\n DMSO 14mM TR']
fit_linear = True
array = labels

#labels are like this XXX-Machine. I want the data to be grouped by the machine

ax1.errorbar(array, df['SSB CONV'], yerr=df['sdSSB CONV'], fmt='o', label='CONV', capsize=capsize, color=color_CONV)
ax1.errorbar(array, df['SSB FLASH'], yerr=df['sdSSB FLASH'], fmt='o', label='FLASH', capsize=capsize, color=color_UHDR)
ax1.set_title('$\\beta_{SSB}$')
if 'Fe' in keywords:
    ax1.set_xlabel('Fe(II) concentration (uM)')
elif 'pH' in keywords:
    ax1.set_xlabel('pH')
else :
    ax1.set_xticklabels(labels, rotation=90)

ax1.grid()
#ax1.set_ylim(0, 0.6)

ax2.errorbar(array, df['DSB CONV'], yerr=df['sdDSB CONV'], fmt='o', label='CONV', capsize=capsize, color=color_CONV)
ax2.errorbar(array, df['DSB FLASH'], yerr=df['sdDSB FLASH'], fmt='o', label='FLASH', capsize=capsize, color=color_UHDR)
ax2.set_title('$\\beta_{DSB}$')
#ax2.set_ylim(0, 0.009)
if 'Fe' in keywords:
    ax2.set_xlabel('Fe(II) concentration (uM)')
elif 'pH' in keywords:
    ax2.set_xlabel('pH')
else :
    ax2.set_xticklabels(labels, rotation=90)

ax2.grid()

error_CONV = (df['SSB CONV']/df['DSB CONV'])*np.sqrt((df['sdSSB CONV']/df['SSB CONV'])**2+(df['sdDSB CONV']/df['DSB CONV'])**2)
error_FLASH = (df['SSB FLASH']/df['DSB FLASH'])*np.sqrt((df['sdSSB FLASH']/df['SSB FLASH'])**2+(df['sdDSB FLASH']/df['DSB FLASH'])**2)

# Define the positions of bars
bar_width = 0.4
array_indices = np.arange(len(array))

# Plot histograms
conv_bars = ax3.bar(array_indices - bar_width/2, df['SSB CONV']/df['DSB CONV'], bar_width, label='CDR (1 Gy/s)', color=color_CONV, alpha=0.5)
flash_bars = ax3.bar(array_indices + bar_width/2, df['SSB FLASH']/df['DSB FLASH'], bar_width, label='FLASH (> 500Gy/s)', color=color_UHDR, alpha=0.5)

# Add error bars
ax3.errorbar(array_indices - bar_width/2, df['SSB CONV']/df['DSB CONV'], yerr=error_CONV, fmt='o', capsize=capsize, color=color_CONV)
ax3.errorbar(array_indices + bar_width/2, df['SSB FLASH']/df['DSB FLASH'], yerr=error_FLASH, fmt='o', capsize=capsize, color=color_UHDR)

vline_positions = [1.5,4.5]  # Example position
for vline_position in vline_positions:
    ax3.axvline(x=vline_position, color='k', linestyle='--', linewidth=1)

# Set labels and legend
ax3.set_ylabel('$\\beta_{SSB}/\\beta_{DSB}$')
ax3.set_xticks(array_indices)
ax3.set_xticklabels(array)
ax3.legend()

plt.tight_layout()
plt.savefig('SSB-DSB.svg', dpi=600)
plt.show()