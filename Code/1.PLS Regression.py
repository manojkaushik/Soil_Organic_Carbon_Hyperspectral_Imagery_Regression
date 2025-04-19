"""
Created on Tue Dec  6 12:57:17 2022
@author: Manoj Kaushik
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


filename = r'..\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')

matirial = 'Org_Car_g_per_kg' # Org_Car_g_per_kg, Clay
column_names_feature = data.columns.difference([matirial])

x = data[column_names_feature].values
y = data[matirial].values

# Set font family
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'Times New Roman'

# Set DPI
plt.rcParams['figure.dpi'] = 150


# # All Bands plotting
# with plt.style.context('bmh'):
#     plt.plot(column_names_feature, x.T)
#     plt.margins(x=0)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance')


def optimise_pls_cv(X, y, n_comp):
    pls = PLSRegression(n_components=n_comp)
    y_cv = cross_val_predict(pls, X, y, cv=10)
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    return (y_cv, r2, mse, rpd)


# you can try full testing with x.shape[1] number of components
r2s = []
mses = []
rpds = []
xticks = np.arange(1, 58)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(x, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)


# Plot the mses
def plot_metrics(vals, ylabel, objective):    
    with plt.style.context('bmh'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('Partial Least Square')
    plt.show()


np.array(mses)[np.argmin(mses)]
np.array(rpds)[np.argmax(rpds)]
np.array(r2s)[np.argmax(r2s)]

xticks[np.argmin(mses)]


plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')


y_cv, r2, mse, rpd = optimise_pls_cv(x, y, 8)
print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))
# R2: 0.6998, MSE: 3.4532, RPD: 1.8252


plt.figure(figsize=(8, 8))
with plt.style.context('bmh'):
    plt.scatter(y, y_cv, color='red', label='Estimated SOC(g/Kg)')
    # plt.plot(y, y, '-g', label='Expected regression line')
    
    # Add text with a bounding box
    x_pos = 8.5  # Adjust position based on data
    y_pos = 19.5  # Adjust position based on data
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    # plt.text(x_pos, y_pos, r'${R^2}$ = 0.71' + '\nMSE = 3.30\nRPD = 1.86', ha='center', va='bottom', bbox=bbox)
    plt.text(x_pos, y_pos, f'${{R^2}}$ = {r2:.2f}\nMSE = {mse:.2f}\nRPD = {rpd:.2f}', ha='center', va='bottom', bbox=bbox)
    
    # Plot regression line
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', linestyle=':', label='Predicted regression line') # 
    
    # Add labels
    plt.xlabel('Measured SOC (g/Kg)', color='royalblue')
    plt.ylabel('Estimated SOC (g/Kg)', color='royalblue')
    
    # Customize legend box
    plt.legend(
        loc='lower right',           # Position of the legend
        # bbox_to_anchor=(1, 1),     # Move legend outside the plot (adjust as needed)
        frameon=True,              # Show legend box
        edgecolor='black',         # Border color of the legend box
        facecolor='lightgray',     # Background color of the legend box
        shadow=True,               # Add shadow to the legend box
        # fontsize='medium',        # Font size of the legend text
        # title='Legend',            # Add a title to the legend
        # title_fontsize='large'     # Font size of the legend title
    )
    plt.plot()


###############-------------------####################

# Experimental results for Sheet14 (Denimme Hyperspectral Data) for SOC
# without wavelet: R2: R2: 0.7128, MSE: 3.3040, RPD: 1.8660
# with    wavelet: R2: R2: 0.7060, MSE: 3.3820, RPD: 1.8443

# Experimental results for Sheet15 (Chilka AVIRIS_NG Data)  for SOC
# without wavelet: R2: 0.6998, MSE: 3.4532, RPD: 1.8252
# with    wavelet: R2: 0.6855, MSE: 3.6176, RPD: 1.7832

# Experimental results for Sheet17 (Denimme Hyperspectral Data)  for Clay content
# without wavelet: R2: 0.9171, MSE: 3.1262, RPD: 3.4736
# with    wavelet: R2: 0.9070, MSE: 3.5092, RPD: 3.2786

# Experimental results for Sheet18 (PRISM Data)  for SOC
# without wavelet: R2: R2: 0.6400, MSE: 4.1414, RPD: 1.6667
# with    wavelet: R2: R2: 0.6331, MSE: 4.2206, RPD: 1.6510































