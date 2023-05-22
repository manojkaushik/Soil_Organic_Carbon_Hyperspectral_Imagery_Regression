# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:08:30 2022
@author: Manoj Kaushik
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.cm as cm
from osgeo import gdal
from osgeo import osr

# Source image
print("\n Input the source image: \n")
tk.Tk().withdraw()
Source_image = filedialog.askopenfilename()
Xs = rasterio.open(Source_image)
Xs = Xs.read()

# Reading File Information
dataset = rasterio.open(Source_image)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)

Xs = np.moveaxis(Xs,0,-1) # moving number of bands at last

# NDVI calculation function
def ndvi(img):
    r=img[:, :, 52]
    nir=img[:, :, 118]
    dinom = (nir+r)
    numer = (nir-r)
    ndvi = np.where(dinom==0.0, 0.0, ((numer/dinom)*1.0))
    plt.imshow(ndvi,cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    return ndvi

Xt_ndvi = ndvi(Xs)

plt.imshow(Xt_ndvi, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()

# Save NDVI Image
path = r"...path\NDVI_n_Mask_Image\part-3"

I_ndvi = (((Xt_ndvi - Xt_ndvi.min()) / (Xt_ndvi.max() - Xt_ndvi.min())) * 255.9).astype(np.uint8)
im = Image.fromarray(I_ndvi)
im.save(path+"\\ndvi.png")

# Creating the mask for soil
def mask(ndvi_image):
    soil_mask = np.empty([Xt_ndvi.shape[0]*Xt_ndvi.shape[1]])
    for i in range(0, len(ndvi_image)) :
        if ndvi_image[i] > 0 and ndvi_image[i] <= 0.3:
            soil_mask[i] = 1
        else:
            soil_mask[i] = 0
    return soil_mask.reshape(Xt_ndvi.shape[0], Xt_ndvi.shape[1])
            
soil_mask = mask(Xt_ndvi.reshape(Xt_ndvi.shape[0]*Xt_ndvi.shape[1]))

plt.imshow(soil_mask, cmap="gray", vmin=-1, vmax=1)
plt.colorbar()

np.unique(soil_mask)

# Save Binary Mask Image
# np.save(path+"\\soil_mask", soil_mask)
plt.imsave(path+"\\soil_mask.png", soil_mask, cmap=cm.gray)


# Overlapping mask one by one on each channel
masked_val = []
for i in range(bands):
    band_array = Xs[:,:,i]
    masked_val.append(np.multiply(band_array, soil_mask))

masked_value_array = np.asanyarray(masked_val)


# rotate channels for just view
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the last
plt.imshow(masked_value_array[:,:,75])

# rerotate the bands for further processing
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the bands
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the bands

# writing the output file
Output_filename = r"...path\Soil_Segmented_image\part-3.tiff"

dataset = gdal.Open(Source_image, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32

driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
output_file = driver.Create(Output_filename, cols, rows, bands, gdal_datatype)
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

for i, image in enumerate(masked_value_array, 1):
        output_file.GetRasterBand(i).WriteArray( image )

prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None



















