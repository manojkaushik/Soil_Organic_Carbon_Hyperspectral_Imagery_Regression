# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 21:43:51 2022
@author: Manoj Kaushik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
# import pywt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

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

# All Bands plotting
with plt.style.context('bmh'):
    plt.plot(column_names_feature, x.T)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')

# Define the kernel (RBF kernel)
kernel = 10.0 * RBF(length_scale=10.0)

def cv_regressor(X, y):
    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    y_cv = cross_val_predict(regressor, X, y, cv=10)
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    return (y_cv, r2, mse, rpd)


y_cv, r2, mse, rpd = cv_regressor(x, y)
print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))
# before grid search: R2: -4.5838, MSE: 64.2355, RPD: 0.4232
# after grid search : 


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
    

#---- Drid search DeepSeek
import numpy as np
from numpy import arange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

param_grid = {
    "kernel__k1__k1__constant_value": arange(0, 10, 0.1),
    "kernel__k1__k2__length_scale": arange(0, 10, 0.1),
    "kernel__k2__noise_level": arange(0, 10, 0.1),
    "n_restarts_optimizer": arange(0, 10, 0.1),
}

grid_search = GridSearchCV(estimator=gpr, param_grid=param_grid, cv=5, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(x, y)

best_gpr = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test MSE: {mse}")

random_search = RandomizedSearchCV(estimator=gpr, param_distributions=param_grid, 
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error', 
                                   n_jobs=-1, random_state=42)
random_search.fit(x, y)
best_gpr = random_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
# Best Parameters: {'kernel__k1__k1__constant_value': 10.0, 'kernel__k1__k2__length_scale': 10.0, 'kernel__k2__noise_level': 10.0}









