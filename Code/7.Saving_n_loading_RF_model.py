# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:36:07 2022

@author: Manoj Kaushik
"""

# saving the ML Regression model for further use by external loading
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# loading data
filename = r'..\Demmin_data.xlsx'
data = pd.read_excel(filename, sheet_name='Sheet15', engine='openpyxl')

column_names_feature = data.columns.difference(['Org_Car_g_per_kg'])

x = data[column_names_feature].values
y = data['Org_Car_g_per_kg'].values

# making of model
model = RandomForestRegressor(bootstrap=True, max_depth=None, max_features='sqrt', 
                                  min_samples_leaf=4, min_samples_split=10, n_estimators=100)
model.fit(x, y)
y_pred = model.predict(x[0].reshape(1, -1))
    
model_path = r"..\SOC Project\Saved_model\Sheet15_338_bands_Chilka_AVIRIS_NG_HSI\RFR_model.sav"
pickle.dump(model, open(model_path, 'wb'))

# loading model for testing again
loaded_model = pickle.load(open(model_path, 'rb'))
result = loaded_model.score(x, y)
print(result) # RF: 0.6449535910060323

