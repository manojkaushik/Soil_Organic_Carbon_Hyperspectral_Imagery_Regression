# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:31:38 2022

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# loading data
filename = r'...path\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')
column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# making of model
pls_model = PLSRegression(n_components = 8)
pls_model.fit(x, y)
y_pred = pls_model.predict(x[0].reshape(1, -1))

# where to save the trainied model
model_path = r"\\...path\Saved_model\Sheet15_338_bands\pls_model.sav"
pickle.dump(pls_model, open(model_path, 'wb'))

# loading model for testing again
loaded_pls_model = pickle.load(open(model_path, 'rb'))
result = loaded_pls_model.score(x, y)
print(result)

# to plot the graph between actual and predicted
y_pred = []
for i in x:
    pred = pls_model.predict(i.reshape(1, -1))
    y_pred.append(pred[0][0])
    
y_pred = np.asarray(y_pred)

plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_pred, color='red')
    plt.plot(y, y, '-g', label='Actual regression line')
    z = np.polyfit(y, y_pred, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()


plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_pred, color='red')
    plt.plot(y, y, '-g', label='Actual regression line')