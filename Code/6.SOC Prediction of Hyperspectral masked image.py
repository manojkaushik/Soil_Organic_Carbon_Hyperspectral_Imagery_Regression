# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:35:36 2022

@author: Manoj Kaushik
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib.cm as cm
import pickle
from tqdm import tqdm
from osgeo import gdal
from osgeo import osr

# load the Masked Image(Segmented Image)
print("\n Input the source image: \n")
tk.Tk().withdraw()
path_source_image = filedialog.askopenfilename()
print(path_source_image)
Xs = rasterio.open(path_source_image)
Xs = Xs.read()

# Reading File Information
dataset = rasterio.open(path_source_image)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)

# normalize the image according to ASD reflactance values between 0 to 1
Xs = np.moveaxis(Xs, 0, -1) # moving number of bands at last
Xs = Xs.reshape(Xs.shape[0]*Xs.shape[1], Xs.shape[2])
Xs = Xs / 10000 # 10000 bcz of ENVI standards

# loading model for testing again
model_path = r"...path\Saved_model\Sheet15_338_bands\PLS_model.sav"

loaded_model = pickle.load(open(model_path, 'rb'))
y_pred = loaded_model.predict(Xs[0].reshape(1, -1))
print(y_pred[0])

# making of SOC Image
soc_pred_img = np.empty([rows*cols])
for i in tqdm(range(Xs.shape[0])):
    if np.any(Xs[i]):
        pred = loaded_model.predict(Xs[i].reshape(1, -1))[0]
        soc_pred_img[i] = pred

soc_pred_img = soc_pred_img.reshape(rows, cols)

# histogram
plt.hist(soc_pred_img)

plt.imshow(soc_pred_img, cmap="CMRmap_r", vmin=0, vmax=150)
plt.colorbar()

# path to save SOC prediction images
path = r"...path\Soil_Segmented_image\part-1"

# Save originaly predicted SOC image
# np.save(path+"\\ori_soc_img", soc_pred_img)
plt.imsave(path+"\\ori_soc_img_gray.png", soc_pred_img, cmap=cm.gray)

# removing the negative values
non_neg_soc_pred_img = np.where(soc_pred_img<0.0, 0, soc_pred_img)

# histogram
plt.hist(non_neg_soc_pred_img)

# plt.imshow(non_neg_soc_pred_img, cmap="CMRmap_r", vmin=0, vmax=50)
# plt.colorbar()

# removing greater than 50
zero_to_50_soc_pred_img = np.where(non_neg_soc_pred_img>50.0, 0, non_neg_soc_pred_img)
plt.hist(zero_to_50_soc_pred_img)


less_than_ten = np.where(non_neg_soc_pred_img>30.0, 0, non_neg_soc_pred_img)
plt.hist(less_than_ten)
plt.imshow(less_than_ten, cmap="CMRmap_r", vmin=0, vmax=150)
plt.colorbar()

# Save adjusted predicted SOC image
# np.save(path+"\\zero_to_50_soc_img", zero_to_50_soc_pred_img)

fig = plt.figure(num=None, figsize=(16, 16), dpi=200, facecolor='w', edgecolor='red')

plt.imshow(zero_to_50_soc_pred_img, cmap="CMRmap_r")
plt.colorbar()
# plt.axis('off')
plt.savefig(path+"\\zero_to_50_soc_img_index11.png",
            bbox_inches="tight",
            dpi=200,
            # pad_inches=0.3,
            # transparent=False
            )


# saving gray image
plt.imsave(path+"\\zero_to_50_soc_img_gray.png", zero_to_50_soc_pred_img, cmap=cm.gray)


# saving SOC predicted image as tiff with geo cordinates
Output_filename = r"...path\Soil_Segmented_image\part-1\part-1_SOC.tiff"

dataset = gdal.Open(path_source_image, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
output_file = driver.Create(Output_filename, cols, rows, 1, gdal_datatype)
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

for i, image in enumerate([zero_to_50_soc_pred_img], 1):
    output_file.GetRasterBand(i).WriteArray(image)

prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None










