# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:58:45 2022
@author: Manoj Kaushik
"""

# import gdal
# from gdal import *
# from gdal import osr
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from rasterio.plot import show

#input the file
filename =  r"..\Hyspex\L2_HySPEX_Demmin_mosaic_ost.bsq"

#Reading File Information
dataset = rasterio.open(filename)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)

with rasterio.open(filename) as r:
    raster_matrix_image = r.read()

blue = raster_matrix_image[11,:,:]
green = raster_matrix_image[30,:,:]
red = raster_matrix_image[63,:,:]

image = [red, green, blue]

type(image)
type(red)
type(green)
type(blue)

rgb = np.dstack((red, green, blue))

plt.figure()
plt.imshow(rgb)

view = imshow(rgb)

np.min(rgb)
np.max(rgb)

# Function to normalize the grid values
def normalize(array):
 """Normalizes numpy arrays into scale 0.0 - 1.0"""
 array_min, array_max = array.min(), array.max()
 return ((array - array_min)/(array_max - array_min))


n_red = normalize(red)
n_green = normalize(green)
n_blue = normalize(blue)
n_rgb = np.dstack((n_red, n_green, n_blue))

plt.figure()
plt.imshow(n_rgb)

view = imshow(n_rgb)

print("Histogram:")
plt.hist(rgb.ravel(), 1750, [1, 1750])
plt.show()

show(n_rgb)
















































