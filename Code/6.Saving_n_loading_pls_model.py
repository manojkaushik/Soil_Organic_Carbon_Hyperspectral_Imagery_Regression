# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:31:38 2022

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

# loading data
filename = r'..\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')

matirial = 'Org_Car_g_per_kg' # Clay
column_names_feature = data.columns.difference([matirial])

x = data[column_names_feature].values
y = data[matirial].values

# making of model
pls_model = PLSRegression(n_components = 8)
pls_model.fit(x, y)
y_pred = pls_model.predict(x[0].reshape(1, -1))

# where to save the loaded model
model_path = r"..\SOC Project\Saved_model\Sheet15_338_bands_Chilka_AVIRIS_NG_HSI\PLSR_model.sav"
pickle.dump(pls_model, open(model_path, 'wb'))

# loading model for testing again
loaded_model = pickle.load(open(model_path, 'rb'))
result = loaded_model.score(x, y)
print(result) # 0.8352299966970731


























