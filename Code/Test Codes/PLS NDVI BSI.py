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
# NDVI: https://developers.planet.com/docs/planetschool/calculate-an-ndvi-in-python/

from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import rasterio

filename = r'..\SOC Project\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet9', engine='openpyxl')

b_bands = pd.Series(np.arange(450, 496, 1))
g_bands = pd.Series(np.arange(495, 571, 1))
r_bands = pd.Series(np.arange(620, 751, 1))
nir_bands = pd.Series(np.arange(750, 1101, 1))
short_wave_IR_bands = pd.Series(np.arange(1550, 1750, 1))
mid_IR_bands = pd.Series(np.arange(2080, 2350, 1))

# column_names_feature = pd.concat([b, g, r], axis=0)

blue = data[b_bands].values
green = data[g_bands].values
red = data[r_bands].values
nir = data[nir_bands].values
short_wave_IR = data[short_wave_IR_bands].values
mid_IR = data[mid_IR_bands].values

b = np.mean(blue, axis=1)
g = np.mean(green, axis=1)
r = np.mean(red, axis=1)
nir = np.mean(nir, axis=1)
swir = np.mean(short_wave_IR, axis=1)
midir = np.mean(mid_IR, axis=1)

# Allow division by zero
np.seterr(divide='ignore', invalid='ignore')

###### Vegitation indices ######
# nir.astype(float)

# Normalized Difference Vegetation Index (NDVI)
ndvi = (nir - r) / (nir + r)

# Soil-Adjusted Vegetation Index (SAVI) 
savi = 1.5 * ((nir - r)/(nir + r + 1.5))

# Optimized Soil Adjusted Vegetation Index (OSAVI)
osavi = (nir - r) / (nir + r + 0.16)

# Modified Soil Adjusted Vegetation Index (MSAVI)
msavi = 0.5 * (2*nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - r)))

# Renormalized Difference Vegetation Index(RDVI)
rdvi = (nir - r)/(np.sqrt(nir + r))

# Differenced Vegetation Index (DVI)
dvi = nir - r

# Soil and Atmospherically Resistant Vegetation Index (SARVI)

###### Soil indices ########

# Combined Spectral Response Index (COSRI)
cosri = ((b + g)/(r + nir)) * ndvi

# Brightness Index(BI)
bi = np.sqrt(r**2 + g**2)

# Clay Index (CI)
ci = swir/midir

# Carbonate Index (car_i)
car_i = r/g

# A New Index for Remote Sensing of Soil Organic Carbon Based Solely on Visible Wavelengths
soci = b/(r*g)


###### Other indices ########
# bsi = 100 + (100 * (((r + g)-(r + b))/((nir + g) + (r + b))))


# ndvi, savi, osavi, msavi, rdvi, dvi, cosri, bi, ci, car_i, soci
x = np.vstack((osavi, bi, ci))
x = x.T
y = data['Org_Car_g_per_kg'].values

print(x.shape, y.shape)

# Plot the data
wl = np.arange(1, x.shape[1]+1, 1)
print(len(wl), wl.shape)
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Reflectance")

x = savgol_filter(x, 3, polyorder=2, deriv=1)

# let's, plot and see
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Reflectance")


def optimise_pls_cv(X, y, n_comp):
    pls = PLSRegression(n_components=n_comp)
    y_cv = cross_val_predict(pls, X, y, cv=10)

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

plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')

y_cv, r2, mse, rpd = optimise_pls_cv(x, y, 4)
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
#  NDVI, SAVI, MSAVI, SARVI, RDVI, and DVI

# Experimental results

# With B, G, R, NIR
# R2: 0.3533, MSE: 7.4392, RPD: 1.2435

# With B, G, R, NIR savgol filter:
# R2: 0.4182, MSE: 6.6924, RPD: 1.3111
#
#  With ndvi, soci
# R2: 0.4069, MSE: 6.8233, RPD: 1.2985
# 
# With soci only
# R2: 0.4499, MSE: 6.3280, RPD: 1.3483

# ndvi, savi, osavi, msavi, rdvi, soci
# R2: 0.3109, MSE: 7.9270, RPD: 1.2047

# ndvi, savi, osavi, msavi, rdvi, soci with derivateive 1
# R2: 0.3131, MSE: 7.9024, RPD: 1.2065

# Second derivative of: ndvi, savi, osavi, msavi, rdvi, dvi, cosri, bi, ci, car_i, soci
# R2: 0.2416, MSE: 8.7250, RPD: 1.1483

# osavi, bi, ci
# R2: 0.3305, MSE: 7.7015, RPD: 1.2222
















