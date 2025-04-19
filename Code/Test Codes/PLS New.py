# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:39:00 2022
@author: USER

Demmin	Original Data
Sheet1	Total 63 samples with different measurments
Sheet2	62 samples with all 2151 wavelengths
Sheet3	62 samples with all 2151 wavelengths with SOC and groupings for classification
Sheet4	62 samples with all 2151 wavelengths with grouping only for classification
Sheet5	62 samples with 401 bands; rangining from 350 - 750 nm for grouping classification
Sheet6	62 samples with 401 bnads; ranging from 350 - 750 nm for SOC Regdression
Sheet7	57 samples (5 samples are removed based on very high organic content) with 401 bands; ranging from 350 - 750 nm for SOC regression
Sheet8	62 samples with 1488 bands; ranging from 400 - 2349 nm for SOC regression
Sheet9	57 samples with all 2151 bands; rangining from 350 - 2500 nm for SOC regression
"""

# https://www.kaggle.com/code/phamvanvung/partial-least-squares-regression-in-python

from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

filename = r'..\SOC Project\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet9', engine='openpyxl')

# column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

a = pd.Series(np.arange(1890, 1925, 1))
# b = pd.Series(np.arange(1641, 1790, 1))
# c = pd.Series(np.arange(1961, 2350, 1))

# column_names_feature = pd.concat([a, b, c], axis=0)

x = data[a].values
y = data['Org_Car_g_per_kg'].values

# Plot the data
wl = np.arange(1, x.shape[1]+1, 1)
print(len(wl), wl.shape)
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Reflectance")

x2 = savgol_filter(x, 17, polyorder=2, deriv=2)

# let's, plot and see
with plt.style.context('ggplot'):
    plt.plot(wl, x2.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Reflectance")
    

def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)


# test with 40 components
r2s = []
mses = []
rpds = []
xticks = np.arange(1, 35)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(x2, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)


# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')

y_cv, r2, mse, rpd = optimise_pls_cv(x2, y, 4)
print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))


plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_cv, color='red')
    plt.plot(y, y, '-g', label='Expected regression line')
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()


# further explore "Variable selection method for PLS in Python": https://nirpyresearch.com/variable-selection-method-pls-python/
























