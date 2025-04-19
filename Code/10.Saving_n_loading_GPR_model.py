# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:36:07 2025

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# loading data
filename = r'..\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# making of model
# Define the kernel (RBF kernel)
kernel = 1.0 * RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
model.fit(x, y)
y_pred = model.predict(x[0].reshape(1, -1))
    
model_path = r"..\SOC Project\Saved_model\Sheet15_338_bands_Chilka_AVIRIS_NG_HSI\GPR_model.sav"
pickle.dump(model, open(model_path, 'wb'))

# loading model for testing again
loaded_model = pickle.load(open(model_path, 'rb'))
result = loaded_model.score(x, y)
print(result) # 1.0
