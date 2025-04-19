# -*- coding: utf-8 -*-
"""
Created on Feb 22 09:47:00 2023

@author: Manoj Kaushik

NOTE: this code is note tested properly...
do spectral smoothing of hyperspectral image using ENVI software by following: spectral> THOR workflows > Tools > Spectral Smmothing with [9, 0, 1] parameters
"""


# from future import absolute_import
# from Py6S import *
import rasterio
import numpy as np
import os
from scipy.signal import savgol_filter
from osgeo import gdal, osr

from tkinter import filedialog as fd

from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs


filename= fd.askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
dirpath = fd.askdirectory()# where to save the image
os.chdir(dirpath)


dataset = rasterio.open(filename)
print(dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)
trans = dataset.transform

with rasterio.open(filename) as r:
    data_matrix = r.read()

new_data_matrix = data_matrix.reshape(bands,(rows*cols) )
    
#p1= new_data_matrix[:,150]
#import matplotlib.pyplot as plt
#plt.plot(p1,'r')  
    

coeffs= savgol_coeffs(9, 1)


for i in range(rows*cols):
    data=new_data_matrix[:,i]
#    print str(i)+str('pixels corrected')
    data[np.isnan(data)] = 0  
    smoothed_data= convolve1d(data, coeffs)
#    np._fit_edges_polyfit(data,9,1,smoothed_data)
#    smoothed_data=savgol_filter(data,9,1,mode='interp')
    new_data_matrix[:,i] = smoothed_data


output_data_matrix = np.array(new_data_matrix)  
output_data_matrix = output_data_matrix.reshape(bands,rows,cols) 
dst_filename = str(filename)+'smoothed_product.tif'
dataset = gdal.Open(filename, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 
dst_ds = driver.Create( dst_filename, output_data_matrix.shape[2], output_data_matrix.shape[1], output_data_matrix.shape[0], gdal_datatype)
dst_ds.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))    

for i, image in enumerate(output_data_matrix, 1):
        dst_ds.GetRasterBand(i).WriteArray( image )

prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
dst_ds.SetProjection(outRasterSRS.ExportToWkt())
dst_ds.FlushCache()
dst_ds = None

    

    
    