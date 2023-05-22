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
import pywt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# loading data
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

def cv_regressor(X, y):
    # regressor = SVR(kernel = 'rbf')
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    y_cv = cross_val_predict(regressor, X, y, cv=10)
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    return (y_cv, r2, mse, rpd)

y_cv, r2, mse, rpd = cv_regressor(x, y)
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