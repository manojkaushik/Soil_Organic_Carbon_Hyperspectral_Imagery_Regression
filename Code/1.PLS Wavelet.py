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
import pywt

filename = r'...path\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# All Bands plotting
with plt.style.context('ggplot'):
    plt.plot(column_names_feature, x.T)

# Wavelet decomposition
(cA, cD) = pywt.dwt(x, 'db6')
x = cA

wl = np.arange(1, x.shape[1]+1, 1)
print(len(wl), wl.shape)

# Plot the data
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Reflectance")

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

np.array(mses)[np.argmin(mses)]
np.array(rpds)[np.argmax(rpds)]
np.array(r2s)[np.argmax(r2s)]

xticks[np.argmin(mses)]
plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')

y_cv, r2, mse, rpd = optimise_pls_cv(x, y, 8)
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