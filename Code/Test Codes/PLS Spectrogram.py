# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:41:39 2022
@author: manoj

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

# Will try some other time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import rasterio

# Audio
import librosa.display, librosa
from librosa.util import normalize as normalize
import IPython.display as ipd

filename = r'..\SOC Project\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet9', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])[100:2051]

# MAnoj band suggestions
# [450-2400] bands to indexes [100:2051]  | working very fine
# [1300-2300] bands to indexes [950:1950] | not so good

# MAnohar band suggestions in which water absorption bands are not there
# [1641 - 1790] bands to indcies => [1291:1441]
# [1890 - 1925] bands to indcies => [1540:1576]
# [1961 - 2350] bands to indcies => [1611:2001]

# SOC bands taken from => https://doi.org/10.3390/rs11111298
# VNIR bands
# [433-435] bands to indcies => [83:86]
# [711-715] bands to indcies => [361:366]
# [727]     bands to indcies => [377:378]
# [986-998] bands to indcies => [636:649]
# NIR-SWIR band range
# [2365-2373] bands to indcies => [2015:2024]
# [2481-2500] bands to indcies => [2131:2151]
# [2198-2206] bands to indcies => [1848:1857]


# feature_1 = column_names_feature[83:86]
# feature_2 = column_names_feature[361:366]
# feature_3 = column_names_feature[377:378]
# feature_4 = column_names_feature[636:649]
# feature_5 = column_names_feature[1848:1857]
# feature_6 = column_names_feature[2015:2024]
# feature_7 = column_names_feature[2131:2151]

# column_names_feature = feature_1.append(feature_2).append(feature_3).append(feature_4).append(feature_5).append(feature_6).append(feature_7)

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# scatter plot for each band
for i in range(58):
    plt.scatter(column_names_feature, x[i, :])
    plt.show()

# All Bands plotting
with plt.style.context('ggplot'):
    plt.plot(column_names_feature, x.T)

# Will try some other time
 
# Signal Processing Parameters
fs = 44100         # Sampling Frequency
n_fft = 2048      # length of the FFT window
hop_length = 812   # Number of samples between successive frames
testset_size = 0.25 # Percentage of data for Testing

wl = np.arange(1, x.shape[1]+1, 1)
print(len(wl), wl.shape)

# Plot the data
with plt.style.context('ggplot'):
    plt.plot(wl, x.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Reflectance")

# x = savgol_filter(x, 3, polyorder=2, deriv=1)

# # let's, plot and see
# with plt.style.context('ggplot'):
#     plt.plot(wl, x.T)
#     plt.xlabel("Wavelengths (nm)")
#     plt.ylabel("D2 Reflectance")


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

###############-------------------####################
# When applied wavlet decomposition with db1 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5726, MSE: 4.9166, RPD: 1.5297

# When applied wavlet decomposition with db2 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5887, MSE: 4.7319, RPD: 1.5592

# When applied wavlet decomposition with db3 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5873, MSE: 4.7482, RPD: 1.5565

# When applied wavlet decomposition with db4 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5899, MSE: 4.7178, RPD: 1.5615

# When applied wavlet decomposition with db5 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5853, MSE: 4.7711, RPD: 1.5528

# When applied wavlet decomposition with db6 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6027, MSE: 4.5702, RPD: 1.5866

# When applied wavlet decomposition with db7 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5956, MSE: 4.6523, RPD: 1.5725

# When applied wavlet decomposition with db8 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5963, MSE: 4.6445, RPD: 1.5738

# When applied wavlet decomposition with db9 having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6062, MSE: 4.5306, RPD: 1.5935

# When applied wavlet decomposition with 'haar' having n_comp=8 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5889, MSE: 4.7287, RPD: 1.5597

# When applied wavlet decomposition with 'bior1.3' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5908, MSE: 4.7076, RPD: 1.5632

# When applied wavlet decomposition with 'bior1.5' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5945, MSE: 4.6650, RPD: 1.5704

# When applied wavlet decomposition with 'bior2.2' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5897, MSE: 4.7199, RPD: 1.5612

# When applied wavlet decomposition with 'bior2.4' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6000, MSE: 4.6011, RPD: 1.5812

# When applied wavlet decomposition with 'bior2.6' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6001, MSE: 4.6009, RPD: 1.5813

# When applied wavlet decomposition with 'bior3.1' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5469, MSE: 5.2123, RPD: 1.4856

# When applied wavlet decomposition with 'bior3.3' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5994, MSE: 4.6082, RPD: 1.5800

# When applied wavlet decomposition with 'bior3.5' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5921, MSE: 4.6925, RPD: 1.5657

# When applied wavlet decomposition with 'bior3.7' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5797, MSE: 4.8356, RPD: 1.5424

# When applied wavlet decomposition with 'bior4.4' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5914, MSE: 4.7000, RPD: 1.5645

# When applied wavlet decomposition with 'bior5.5' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5953, MSE: 4.6559, RPD: 1.5719

# When applied wavlet decomposition with 'coif1' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5909, MSE: 4.7063, RPD: 1.5635

# When applied wavlet decomposition with 'coif2' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5931, MSE: 4.6804, RPD: 1.5678

# When applied wavlet decomposition with 'coif3' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5913, MSE: 4.7011, RPD: 1.5643

# When applied wavlet decomposition with 'coif4' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6043, MSE: 4.5517, RPD: 1.5898

# When applied wavlet decomposition with 'coif5' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.6019, MSE: 4.5797, RPD: 1.5849

# When applied wavlet decomposition with 'sym2' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5887, MSE: 4.7319, RPD: 1.5592

# When applied wavlet decomposition with 'sym3' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5873, MSE: 4.7482, RPD: 1.5565

# When applied wavlet decomposition with 'sym4' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5922, MSE: 4.6914, RPD: 1.5659

# When applied wavlet decomposition with 'sym5' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5942, MSE: 4.6679, RPD: 1.5699

# When applied wavlet decomposition with 'sym6' having n_comp=9 | with all 2151 bands (range 350 - 2500 nm)
# R2: 0.5953, MSE: 4.6559, RPD: 1.5719

# ************* When applied on selected bands 1950 bands ranging from 451 - 2400 nm *******************

# When applied wavlet decomposition with 'cofi4' having n_comp=8 with 1950 bands (range 451 - 2400 nm)
# R2: 0.7046, MSE: 3.3983, RPD: 1.8399

# When applied wavlet decomposition with 'db6' having n_comp=8 with 1950 bands (range 451 - 2400 nm) $$$$$$$$$$$$$$$$$$$$
# R2: 0.7097, MSE: 3.3401, RPD: 1.8558

# After MAnohar water band removal suggestion | wavlet decomposition with 'db6' having n_comp=5
# R2: 0.6392, MSE: 4.1506, RPD: 1.6648

# VNIR and NIR-SWIR bands selected from https://doi.org/10.3390/rs11111298 
# R2: 0.5740, MSE: 4.9010, RPD: 1.5321 | normal with 'db6' having n_comp=30
# R2: 0.3606, MSE: 7.3552, RPD: 1.2506 | wavlet decomposition with 'db6' having n_comp=10































